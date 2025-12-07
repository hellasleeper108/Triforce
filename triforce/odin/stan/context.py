import asyncio
import logging
import uuid
import time
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from triforce.odin.stan.memory import MemoryEngine, MemoryType
from triforce.odin.stan.ai_provider import AIProvider, ProviderConfig

# --- Data Models ---

class SessionEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    event_type: str # "command", "system_alert", "inference"
    content: str
    metadata: Dict[str, Any] = {}

class ContextSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = "default_user"
    start_time: float = Field(default_factory=time.time)
    active_project: Optional[str] = None
    focus_topic: Optional[str] = None
    history: List[SessionEvent] = []
    autonomy_level: int = 1 # 1=Advisor, 2=Engineer...
    metadata: Dict[str, Any] = {}
    
    def add_event(self, type: str, content: str, meta: Dict = {}):
        self.history.append(SessionEvent(event_type=type, content=content, metadata=meta))
        # Keep history trimmer
        if len(self.history) > 50:
            self.history.pop(0)

# --- Context Manager ---

class ContextManager:
    """
    Manages active sessions and persists episodic memory.
    """
    
    def __init__(self, memory_engine: MemoryEngine, ai_provider: AIProvider):
        self.memory = memory_engine
        self.ai = ai_provider
        self.logger = logging.getLogger("stan.context")
        self.active_sessions: Dict[str, ContextSession] = {}

    def start_session(self, user_id: str = "default_user", project: str = None) -> ContextSession:
        session = ContextSession(user_id=user_id, active_project=project)
        self.active_sessions[session.session_id] = session
        self.logger.info(f"Session {session.session_id} started for {user_id}.")
        return session

    def get_session(self, session_id: str) -> Optional[ContextSession]:
        return self.active_sessions.get(session_id)

    def update_context(self, session_id: str, command: str, result: str):
        session = self.get_session(session_id)
        if not session:
            return
        
        session.add_event("command", command)
        session.add_event("result", result[:200] + "..." if len(result) > 200 else result)
        
        # Simple heuristic to update focus
        if "deploy" in command.lower():
            session.focus_topic = "Deployment"
        elif "debug" in command.lower() or "error" in result.lower():
            session.focus_topic = "Debugging"

    async def end_session(self, session_id: str):
        session = self.get_session(session_id)
        if not session:
            return

        self.logger.info(f"Ending session {session_id}. Generating summary...")
        
        # 1. Generate Summary via AI
        summary = await self._generate_summary(session)
        
        # 2. Store in Memory
        await self.memory.add_memory(
            MemoryType.EPHEMERAL, # Or LONG_TERM if high significance
            text=f"Session Summary ({session.start_time}): {summary}",
            metadata={"session_id": session_id, "project": session.active_project}
        )
        
        # 3. Cleanup
        del self.active_sessions[session_id]
        self.logger.info("Session archived.")

    async def _generate_summary(self, session: ContextSession) -> str:
        history_text = "\n".join([f"[{e.event_type}] {e.content}" for e in session.history[-20:]])
        prompt = f"""
        Summarize this user session for future retrieval.
        Focus on: What occurred? What did the user achieve? Any unresolved issues?
        
        History:
        {history_text}
        """
        try:
            return await self.ai.generate(prompt, system="You are a Scribe. Summarize technical sessions.")
        except Exception:
            return "Session ended (Summary unavailable)."

# --- Demo Driver ---

async def run_context_demo():
    print("--- STAN Context Manager ---")
    
    # 1. Setup Mocks
    from triforce.odin.stan.ai_provider import AIProvider
    class ContextMockAI(AIProvider):
        async def generate(self, prompt, system, **kwargs):
            return "User debugged a VRAM OOM issue on Node Thor. Attempted to deploy Llama-3, failed, then cleaned up processes. Issue resolved."
        async def embed(self, t): return []
        async def classify(self, t, l): return l[0]

    ai = ContextMockAI(ProviderConfig(type="mock"))
    mem = MemoryEngine(ai, db_path=":memory:")
    mgr = ContextManager(mem, ai)
    
    # 2. Start Session
    session = mgr.start_session(user_id="hella", project="Large Inference")
    print(f"Active Session: {session.session_id} (Focus: {session.focus_topic})")
    
    # 3. Simulate Interactions
    mgr.update_context(session.session_id, "deploy llama-3 --node thor", "Error: Out of Memory (OOM)")
    print(f"Updated Focus: {session.focus_topic}")
    
    mgr.update_context(session.session_id, "kill_process --pid 9921", "Process killed. 24GB Freed.")
    mgr.update_context(session.session_id, "deploy llama-3 --node thor", "Success. Model Loaded.")
    
    # 4. End Session & Verify Memory
    await mgr.end_session(session.session_id)
    
    print("\n[Memory Check]")
    mems = await mem.search_memory("What happened in the last session?", limit=1)
    if mems:
        print(f"Retrieved: {mems[0].item.text}")

if __name__ == "__main__":
    asyncio.run(run_context_demo())
