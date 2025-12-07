import logging
import time
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel

# Imports
from triforce.odin.stan.model_routing import ModelRouter, RouteResult
from triforce.odin.stan.model_registry import ModelRegistry

class FailureReport(BaseModel):
    original_model: str
    attempts: int
    final_success: bool
    final_model: Optional[str]
    final_node: Optional[str]
    errors: List[str]
    latency_ms: float

class ReasoningResilienceStrategy:
    """
    The Safety Net for STAN's intelligence.
    Wraps ModelRouter to handle runtime failures (timeouts, crashes, OOMs).
    """

    def __init__(self, router: ModelRouter, registry: ModelRegistry):
        self.router = router
        self.registry = registry
        self.logger = logging.getLogger("stan.resilience")

    async def recover_generate(self, prompt: str, target_model: str, **kwargs) -> Dict[str, Any]:
        """
        Attempts to generate text using the target model.
        If it fails, iterates through the fallback chain and alternative nodes.
        Returns: { "response": str, "report": FailureReport }
        """
        start_ts = time.time()
        errors = []
        
        # 1. Build Candidate List (Target + Fallbacks)
        candidates = [target_model]
        model_def = self.registry.get_model(target_model)
        if model_def:
            candidates.extend(model_def.fallback_chain)
        
        # 2. Iterative Attempt Strategy
        for attempt_idx, model_name in enumerate(candidates):
            self.logger.info(f"Attempt {attempt_idx+1}: Trying {model_name}...")
            
            # Route
            route = self.router.route_generate(prompt, model_name)
            
            if not route.success or not route.selected_node:
                msg = f"Routing failed for {model_name}: {route.reasoning}"
                self.logger.warning(msg)
                errors.append(msg)
                continue
            
            node = route.selected_node
            
            try:
                # Execute
                client = self.router.get_client(node)
                self.logger.debug(f"Sending request to {node.hostname}...")
                
                resp_text = await client.generate(prompt, model=model_name, **kwargs)
                
                # Success!
                elapsed = (time.time() - start_ts) * 1000
                report = FailureReport(
                    original_model=target_model,
                    attempts=attempt_idx + 1,
                    final_success=True,
                    final_model=model_name,
                    final_node=node.hostname,
                    errors=errors,
                    latency_ms=elapsed
                )
                
                if attempt_idx > 0:
                    self.logger.info(f"Recovered using {model_name} on {node.hostname} after {attempt_idx} failures.")
                
                return {"response": resp_text, "report": report}
                
            except Exception as e:
                err_msg = f"{node.hostname} failed on {model_name}: {str(e)}"
                self.logger.error(err_msg)
                errors.append(err_msg)
                
                # Feedback to router (naive)
                # In a real system, we'd mark the node 'suspect' for a short duration
                # self.router.mark_suspect(node.hostname)
        
        # All attempts failed
        elapsed = (time.time() - start_ts) * 1000
        report = FailureReport(
            original_model=target_model,
            attempts=len(errors),
            final_success=False,
            final_model=None,
            final_node=None,
            errors=errors,
            latency_ms=elapsed
        )
        self.logger.critical(f"All recovery attempts failed for {target_model}. Errors: {errors}")
        raise RuntimeError(f"Reasoning Recovery Failed: {errors}")

    async def recover_embed(self, text: str, target_model: str = "nomic-embed-text") -> Dict[str, Any]:
        """
        Resilient embedding generation.
        """
        start_ts = time.time()
        errors = []
        
        # Embedding usually has fewer fallbacks, but we can try other nodes
        model_def = self.registry.get_model(target_model)
        # Verify model exists
        if not model_def:
             # Try generic fallback if specific embedding model missing
             target_model = "nomic-embed-text"
        
        # Simple Retry Loop (Same model, maybe different node if we had dynamic exclusion)
        # For now, we rely on Router finding *a* node.
        # If Router keeps picking the same broken node, we fail.
        # TODO: Implement 'exclude_nodes' in route_request
        
        route = self.router.route_embed(text, target_model)
        if not route.success:
             raise RuntimeError(f"No node found for embedding {target_model}")
             
        try:
             client = self.router.get_client(route.selected_node)
             vector = await client.embed(text, model=target_model)
             return {"vector": vector, "success": True}
        except Exception as e:
             self.logger.error(f"Embedding failed: {e}")
             raise e

# --- Demo Driver ---

if __name__ == "__main__":
    import asyncio
    from triforce.odin.stan.remote_ollama_client import RemoteOllamaClient
    
    # Mock Client that fails on request
    class FlakyClient:
        def __init__(self, fail_count=1):
             self.fail_count = fail_count
             self.calls = 0
             
        async def generate(self, prompt, model, **kwargs):
             self.calls += 1
             if "70b" in model: # Simulate 70b failing
                 raise TimeoutError("Simulated Timeout on 70B")
             return f"Response from {model}"

    # Setup
    registry = ModelRegistry()
    router = ModelRouter(registry)
    
    # Patch Router to return FlakyClient
    # This is a bit dirty for a unit test but works for a script
    router.get_client = lambda node: FlakyClient()

    async def demo():
        print("--- Reasoning Recovery Demo ---")
        strategy = ReasoningResilienceStrategy(router, registry)
        
        print("\n[Scenario] Requesting Llama3 70B (Remote Thor)")
        # We simulate that 70B fails. Fallback chain is [70b -> 8b].
        # 8B should succeed.
        
        try:
            result = await strategy.recover_generate("Analyze", "llama3:70b")
            report = result["report"]
            print(f"Outcome: Success={report.final_success}")
            print(f"Original: {report.original_model}")
            print(f"Final: {report.final_model} (The fallback)")
            print(f"Errors encountered: {report.errors}")
        except Exception as e:
            print(f"Total Failure: {e}")

    asyncio.run(demo())
