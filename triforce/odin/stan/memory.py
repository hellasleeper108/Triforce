import sqlite3
import json
import logging
import math
import time
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel

# Imports
from triforce.odin.stan.ai_provider import AIProvider

# --- Configuration ---
DB_PATH = "stan_memory.db"

class MemoryType(str, Enum):
    EPHEMERAL = "ephemeral"      # Recent tasks/context
    LONG_TERM = "long_term"      # Validated history
    KNOWLEDGE = "knowledge"      # Base model facts/docs
    TELEMETRY = "telemetry"      # Resource patterns
    USER_PREF = "user_pref"      # User preferences

class MemoryItem(BaseModel):
    id: Optional[int]
    type: MemoryType
    text: str
    vector: List[float] # JSON serialized in DB
    metadata: Dict[str, Any]
    timestamp: float

class RetrievalResult(BaseModel):
    item: MemoryItem
    similarity: float

# --- Vector Math Helpers ---

def normalize_vector(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    # Assumes vectors are already normalized for efficiency
    # If not, use dot / (norm*norm)
    # But best practice is to store normalized.
    return sum(a * b for a, b in zip(v1, v2))

# --- Memory Engine ---

class MemoryEngine:
    """
    The Hippocampus of STAN. 
    Stores and retrieves memories using Semantic Vector Search.
    Supports Ollama embeddings (nomic-embed-text typically 768d).
    """

    def __init__(self, ai_provider: AIProvider, db_path: str = DB_PATH):
        self.ai = ai_provider
        self.db_path = db_path
        self.logger = logging.getLogger("stan.memory")
        self._persistent_conn = None
        self._expected_dim = None # Will learn on first insert
        
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            
        self._init_db()

    def _get_conn(self):
        if self._persistent_conn:
            return self._persistent_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                text TEXT,
                vector TEXT, -- JSON list
                metadata TEXT, -- JSON dict
                timestamp REAL
            )
        """)
        conn.commit()
        if not self._persistent_conn:
            conn.close()

    async def add_memory(self, type: MemoryType, text: str, metadata: Dict[str, Any] = {}) -> int:
        """
        Embeds text, normalizes vector, and saves to memory store.
        """
        # 1. Embed
        start_ts = time.time()
        try:
            raw_vector = await self.ai.embed(text)
        except Exception as e:
            self.logger.error(f"Embedding failed for '{text[:20]}...': {e}")
            return -1

        if not raw_vector:
            self.logger.warning("Empty embedding returned. Skipping memory add.")
            return -1

        # 2. Validate Dimensions
        dim = len(raw_vector)
        if self._expected_dim is None:
            self._expected_dim = dim
            self.logger.info(f"Memory initialized with Vector Dim: {dim}")
        elif dim != self._expected_dim:
            self.logger.error(f"Dimension mismatch! Expected {self._expected_dim}, got {dim}. Skipping.")
            return -1

        # 3. Normalize
        vector = normalize_vector(raw_vector)

        # 4. Persist
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO memories (type, text, vector, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
            (type.value, text, json.dumps(vector), json.dumps(metadata), time.time())
        )
        
        mem_id = cursor.lastrowid
        conn.commit()
        if not self._persistent_conn:
            conn.close()
        
        elapsed = (time.time() - start_ts) * 1000
        self.logger.debug(f"Memorized ({type.value}) in {elapsed:.2f}ms")
        return mem_id

    async def search_memory(self, query: str, type_filter: Optional[MemoryType] = None, limit: int = 5) -> List[RetrievalResult]:
        """
        Semantic search for relevant context.
        """
        # 1. Embed Query
        try:
            query_raw = await self.ai.embed(query)
            if not query_raw: return []
            query_vec = normalize_vector(query_raw)
        except Exception as e:
            self.logger.error(f"Search embedding failed: {e}")
            return []

        # 2. Scan & Score 
        conn = self._get_conn()
        cursor = conn.cursor()
        
        sql = "SELECT id, type, text, vector, metadata, timestamp FROM memories"
        params = []
        if type_filter:
            sql += " WHERE type = ?"
            params.append(type_filter.value)
            
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        if not self._persistent_conn:
            conn.close()

        results = []
        for r in rows:
            try:
                vec = json.loads(r[3])
                if len(vec) != len(query_vec): continue # Skip mismatches
                
                # Dot product of normalized vectors = Cosine Similarity
                score = cosine_similarity(query_vec, vec)
                
                item = MemoryItem(
                    id=r[0],
                    type=r[1],
                    text=r[2],
                    vector=vec, # Returning full vector might be heavy, maybe optional?
                    metadata=json.loads(r[4]),
                    timestamp=r[5]
                )
                
                results.append(RetrievalResult(item=item, similarity=score))
            except Exception:
                continue
        
        # 3. Sort & Slice
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    async def delete_memory(self, days_older_than: int = 30):
        """Cleanup logic."""
        cutoff = time.time() - (days_older_than * 86400)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE timestamp < ?", (cutoff,))
        count = cursor.rowcount
        conn.commit()
        if not self._persistent_conn:
            conn.close()
        self.logger.info(f"Pruned {count} old memories.")

    # --- RAG Helper ---

    async def get_context_for_reasoning(self, task_description: str) -> str:
        """
        High-level helper for Brains.
        Returns a formatted string of relevant facts.
        """
        hits = await self.search_memory(task_description, limit=3)
        if not hits:
            return "No relevant past occurrences found."
        
        context_lines = ["Relevant Context:"]
        for h in hits:
            # Filter low relevance
            if h.similarity < 0.6: continue 
            context_lines.append(f"- [{h.similarity:.2f}] {h.item.text}")
        
        if len(context_lines) == 1:
            return "No relevant past occurrences found (low similarity)."
            
        return "\n".join(context_lines)

    async def ingest_log(self, log_entry: Dict[str, Any]):
        """
        Classifies and stores system logs if they are significant.
        """
        msg = log_entry.get("message", "")
        level = log_entry.get("level", "INFO")
        
        if level == "ERROR":
            await self.add_memory(
                MemoryType.LONG_TERM,
                f"Error Log: {msg}",
                metadata={"source": "log", "level": "error"}
            )
        elif "Reflection" in msg:
            await self.add_memory(
                MemoryType.LONG_TERM,
                f"Plan Reflection: {msg}",
                metadata={"source": "reflection"}
            )

# --- Example Usage ---

async def run_memory_demo():
    import random
    from triforce.odin.stan.ai_provider import AIProvider, ProviderConfig, ModelConfig

    # Mock Provider with normalized vectors for testing math
    class NormalizedMockProvider(AIProvider):
        async def embed(self, text):
            # Deterministic pseudo-random for "semantic" simulation
            # "Thor" -> [0.9, 0.1]
            # "Loki" -> [0.1, 0.9]
            if "Thor" in text: return [0.9, 0.1]
            if "Loki" in text: return [0.1, 0.9]
            return [0.5, 0.5]
            
        async def generate(self, p, **k): return ""
        async def classify(self, t, l): return l[0]

    # Setup
    config = ProviderConfig(type="mock") 
    ai = NormalizedMockProvider(config)
    mem = MemoryEngine(ai, db_path=":memory:") 
    
    print("--- Learning Phase ---")
    await mem.add_memory(MemoryType.KNOWLEDGE, "Thor is a GPU node perfect for AI.", {"node": "thor"})
    await mem.add_memory(MemoryType.KNOWLEDGE, "Loki is a weak CPU node.", {"node": "loki"})
    
    # Scenario
    query = "Which node has GPUs?" # Should match Thor
    print(f"\n--- Query: '{query}' ---")
    
    # We cheat the mock: "GPUs" isn't "Thor", so it returns [0.5, 0.5]. 
    # Similarity to [0.9, 0.1] (Thor) vs [0.1, 0.9] (Loki).
    # Normalizing [0.5, 0.5] -> [0.707, 0.707].
    # Thor Norm: [0.99, 0.11].
    # This mock is too simple for real cosine math demo, but proves the pipeline.
    # Let's just run it.
    
    hits = await mem.search_memory(query, limit=2)
    for h in hits:
        print(f"[{h.similarity:.2f}] {h.item.text}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_memory_demo())
