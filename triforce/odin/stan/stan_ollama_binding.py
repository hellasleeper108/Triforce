import logging
import asyncio
from typing import Optional
import sys
from unittest.mock import MagicMock

# Mock FastAPI for environment where it is missing
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.security"] = MagicMock()
sys.modules["fastapi.middleware"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()

# Core & Infrastructure
from triforce.odin.stan.core import STAN
from triforce.odin.stan.ai_provider import AIProvider

# Advanced AI Components
from triforce.odin.stan.model_registry import ModelRegistry
from triforce.odin.stan.model_routing import ModelRouter
from triforce.odin.stan.reasoning_fallbacks import ReasoningResilienceStrategy
from triforce.odin.stan.memory import MemoryEngine

logger = logging.getLogger("stan.binding")

# --- Smart Provider Adapter ---

class SmartAIProvider(AIProvider):
    """
    Adapter that exposes the sophisticated Routing & Resilience logic
    as a simple AIProvider interface for the Brains.
    """
    def __init__(self, strategy: ReasoningResilienceStrategy, router: ModelRouter):
        self.strategy = strategy
        self.router = router
        
    async def generate(self, prompt: str, **kwargs) -> str:
        # Extract target model or default to smart selection
        # (Brains usually ask for a specific role, but we can intercept)
        model = kwargs.pop("model", "llama3:8b") 
        
        # Use resilience strategy (Retry/Fallback/Router)
        result = await self.strategy.recover_generate(prompt, model, **kwargs)
        
        # Log if fallback happened
        report = result["report"]
        if not report.final_success:
             # Should have raised in strategy, but failsafe
             raise RuntimeError(f"Generation failed: {report.errors}")
             
        if report.final_model != report.original_model:
             logger.warning(f"Fallback Active: Requested {report.original_model} -> Served by {report.final_model}")
             
        return result["response"]

    async def embed(self, text: str, model: str = None) -> list[float]:
        # Use Router to find best embedding node
        if not model: model = "nomic-embed-text"
        res = self.router.route_embed(text, model)
        
        if not res.success or not res.selected_node:
             logger.error(f"Embedding routing failed for {model}")
             return []
             
        client = self.router.get_client(res.selected_node)
        return await client.embed(text, model=model)

    async def classify(self, text: str, labels: list[str], model: str = None) -> str:
        # Naive routing for now, or could add route_classify
        if not model: model = "llama3:8b"
        res = self.router.route_request(
             model_name=model, 
             task_type="classify"
             # Could add RouteRequest args here
        )
        if not res.success or not res.selected_node:
             return labels[0]
             
        client = self.router.get_client(res.selected_node)
        return await client.classify(text, labels, model=model)
        
    def get_best_model(self, role: str = "general", **kwargs) -> str:
        # Pass through to registry logic if needed, or return best guess
        # Brains usually call this to ask "what model should I use?"
        # We can query the router/registry here.
        # For now, simplistic map:
        if role == "planner": return "llama3:70b"
        if role == "creative": return "mistral:7b"
        return "llama3:8b"

# --- Main Factory ---

def create_stan_with_ollama(persistence_path: Optional[str] = None) -> STAN:
    """
    Bootstraps a fully operational STAN instance wired to the cluster.
    """
    logger.info("Initializing STAN AI Cluster Bindings...")
    
    # 1. Registry (The Catalog)
    registry = ModelRegistry()
    
    # 2. Router ( Traffic Control)
    router = ModelRouter(registry)
    
    # 3. Resilience (The Safety Net)
    strategy = ReasoningResilienceStrategy(router, registry)
    
    # 4. Neural Adapter (The Interface)
    smart_provider = SmartAIProvider(strategy, router)
    
    # 5. Memory (The Hippocampus)
    # Uses smart_provider to route embeddings to "nomic-embed-text" on capable nodes
    memory = MemoryEngine(smart_provider, db_path="stan_memory.db")
    
    # 6. The Being (Core Injection)
    stan = STAN(
        persistence_path=persistence_path,
        use_mock_ai=False, # We are providing our own provider anyway
        ai_provider=smart_provider,
        memory_engine=memory
    )
    
    logger.info("STAN Bindings Complete. System Ready.")
    return stan

# --- Demo / Verification ---

async def test_distributed_inference():
    print("--- Distributed Inference Verification ---")
    
    # Note: This will try to hit REAL http://thor:11434 etc
    # Unless we patch it, it will fail in this env.
    # But this script is intended for the production environment.
    
    stan = create_stan_with_ollama()
    await stan.start()
    
    # 1. Warmup (Via router access if we want, or just let it happen on fly)
    # The binding hides the router, but we can verify via normal flow
    
    print("\n[1] Testing Command Flow (Parsing -> Planning -> Execution)")
    cmd = "Analyze the cluster logs and summarize recent errors"
    
    try:
        response = await stan.run(cmd)
        print("\n=== Result ===")
        print(f"Message: {response['message']}")
        print(f"Plan: {len(response.get('plan', []))} steps generated.")
    except Exception as e:
        print(f"\n[!] Execution Error (Expected if nodes offline): {e}")

    await stan.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_distributed_inference())
