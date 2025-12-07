import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# Imports
from triforce.odin.stan.model_routing import ModelRouter
from triforce.odin.stan.model_registry import ModelRegistry

class ModelCacheState(BaseModel):
    node_id: str
    model_name: str
    last_loaded: float
    load_time_ms: float
    status: str = "warmed" # warmed, failed

class ModelWarmupService:
    """
    Ensures critical models are loaded in VRAM before users need them.
    """

    def __init__(self, router: ModelRouter, registry: ModelRegistry):
        self.router = router
        self.registry = registry
        self.logger = logging.getLogger("stan.warmup")
        self.cache_state: Dict[str, ModelCacheState] = {} # Key: "{node_id}:{model_name}"

    async def warm_model(self, model_name: str, force: bool = False) -> Optional[ModelCacheState]:
        """
        Routes a dummy request to the best node for this model to trigger loading.
        """
        # 1. Route
        # We assume 'generate' task type for warming
        route = self.router.route_generate(prompt="ping", model_name=model_name)
        
        if not route.success or not route.selected_node:
            self.logger.warning(f"Cannot warm {model_name}: No capable node found.")
            return None
            
        node = route.selected_node
        # Note: Route might have selected a fallback model if primary is unavailable?
        # Actually route_generate returns the *selected* model. 
        # If we asked for 70b and got 8b fallback, we are technically warming 8b.
        target_model = route.selected_model 
        
        cache_key = f"{node.node_id}:{target_model}"
        
        # Check if recently warmed
        if not force and cache_key in self.cache_state:
            last = self.cache_state[cache_key]
            if time.time() - last.last_loaded < 300: # 5 mins TTL for "warm" definition
                self.logger.info(f"{target_model} on {node.hostname} is already warm.")
                return last

        self.logger.info(f"Warming {target_model} on {node.hostname}...")
        start_ts = time.time()
        
        try:
            client = self.router.get_client(node)
            # Send minimal request to force load
            await client.generate(
                prompt="ignore this", 
                model=target_model, 
                max_tokens=1
            )
            
            elapsed = (time.time() - start_ts) * 1000
            
            state = ModelCacheState(
                node_id=node.node_id,
                model_name=target_model,
                last_loaded=time.time(),
                load_time_ms=elapsed
            )
            self.cache_state[cache_key] = state
            self.logger.info(f"Warmed {target_model} on {node.hostname} in {elapsed:.0f}ms")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to warm {target_model} on {node.hostname}: {e}")
            return None

    async def warm_all_primary_models(self):
        """
        Iterates through the registry and warms high-priority models.
        """
        self.logger.info("Starting cluster-wide model warmup...")
        
        # Prioritize based on some heuristic or tags
        # For now, warm everything marked 'medium' or 'large' as they take time
        # Or specifically keys we know
        targets = ["llama3:8b", "llama3:70b", "mistral:7b"]
        
        results = []
        for t in targets:
            res = await self.warm_model(t)
            if res:
                results.append(res)
        
        self.logger.info(f"Warmup complete. {len(results)}/{len(targets)} models ready.")

    def report_model_cache(self) -> List[Dict[str, Any]]:
        return [s.model_dump() for s in self.cache_state.values()]

# --- Demo ---

if __name__ == "__main__":
    import asyncio
    
    # Mocks for standalone run
    from triforce.odin.stan.remote_ollama_client import RemoteOllamaClient
    
    # Mock Client to simulate latency
    class MockClient:
        def __init__(self, *args, **kwargs): pass
        async def generate(self, *args, **kwargs):
            await asyncio.sleep(0.5) # Simulate network + VRAM load
            return "pong"
    
    # Patch Router to return MockClient
    ModelRouter.get_client = lambda self, node: MockClient()
    
    async def demo():
        print("--- Model Warmup Demo ---")
        registry = ModelRegistry()
        router = ModelRouter(registry)
        service = ModelWarmupService(router, registry)
        
        # 1. Warm specific
        await service.warm_model("llama3:70b")
        
        # 2. Warm all
        await service.warm_all_primary_models()
        
        # 3. Report
        print("\nCache Report:")
        for record in service.report_model_cache():
            print(record)

    asyncio.run(demo())
