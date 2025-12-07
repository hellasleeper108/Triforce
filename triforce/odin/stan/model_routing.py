import logging
import random
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

# Imports
from triforce.odin.stan.model_registry import ModelRegistry, ModelDefinition
from triforce.odin.stan.remote_ollama_client import RemoteOllamaClient
from triforce.odin.stan.ai_provider import AIProvider

# --- Data Models ---

class NodeCapability(BaseModel):
    node_id: str
    hostname: str
    is_online: bool
    has_gpu: bool
    vram_gb: float
    ram_gb: float
    current_load: float = 0.0 # 0.0 - 1.0
    supported_models: List[str] = []
    base_url: str # Ollama endpoint, e.g. http://thor:11434

class RouteRequest(BaseModel):
    model_name: str
    task_type: str = "generate" # generate, embed, classify
    priority: int = 1 # 1 (Low) - 10 (Critical)

class RouteResult(BaseModel):
    success: bool
    selected_node: Optional[NodeCapability]
    selected_model: str 
    api_url: Optional[str]
    reasoning: str

# --- Routing Engine ---

class ModelRouter:
    """
    Traffic Controller for STAN's AI requests.
    Directs prompts to the most suitable available node.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.logger = logging.getLogger("stan.router")
        self._client_cache: Dict[str, RemoteOllamaClient] = {}
        
        # In a real system, this comes from the Service Registry / Discovery
        # For now, we hardcode the known cluster topology
        self.nodes: Dict[str, NodeCapability] = {
            "thor": NodeCapability(
                node_id="n1", hostname="thor", is_online=True,
                has_gpu=True, vram_gb=80.0, ram_gb=512.0,
                base_url="http://thor:11434",
                supported_models=["llama3:70b", "llama3:8b", "mistral:7b", "qwen2.5-math:7b"]
            ),
            "odin": NodeCapability(
                node_id="n2", hostname="odin", is_online=True,
                has_gpu=False, vram_gb=0.0, ram_gb=64.0,
                base_url="http://odin:11434",
                supported_models=["llama3:8b", "phi3:3.8b", "qwen2.5-math:7b", "nomic-embed-text"]
            ),
            "loki": NodeCapability(
                node_id="n3", hostname="loki", is_online=True,
                has_gpu=False, vram_gb=0.0, ram_gb=16.0, # MacBook
                base_url="http://loki:11434",
                supported_models=["mistral:7b", "phi3:3.8b", "nomic-embed-text"]
            )
        }

    def get_client(self, node: NodeCapability) -> AIProvider:
        """
        Factory method to get an initialized AIProvider for the target node.
        Uses caching to maintain connection pools.
        """
        if node.node_id in self._client_cache:
            return self._client_cache[node.node_id]
        
        self.logger.info(f"Creating new RemoteOllamaClient for {node.hostname} ({node.base_url})")
        client = RemoteOllamaClient(
            base_url=node.base_url,
            node_id=node.hostname
        )
        self._client_cache[node.node_id] = client
        return client

    async def close_all(self):
        for c in self._client_cache.values():
            await c.close()
        self._client_cache.clear()

    def update_node_status(self, hostname: str, is_online: bool, load: float = 0.0):
        """Called by telemetry system to update routing table."""
        if hostname in self.nodes:
            self.nodes[hostname].is_online = is_online
            self.nodes[hostname].current_load = load

    def route_request(self, req: RouteRequest) -> RouteResult:
        """
        Main entry point. Finds best node + model combo.
        """
        # 1. Look up Model Definition
        model_def = self.registry.get_model(req.model_name)
        if not model_def:
            self.logger.error(f"Unknown model requested: {req.model_name}")
            return RouteResult(success=False, selected_node=None, selected_model=req.model_name, api_url=None, reasoning="Unknown Model")

        # 2. Try Primary Model on Preferred Nodes
        node = self._find_node_for_model(model_def)
        if node:
            return RouteResult(
                success=True,
                selected_node=node,
                selected_model=model_def.model_name,
                api_url=node.base_url,
                reasoning=f"Mapped {model_def.model_name} to preferred node {node.hostname}"
            )

        # 3. Fallback Chain (Downshifting)
        self.logger.warning(f"Primary model {req.model_name} unavailable. Attempting fallbacks...")
        
        for fb_name in model_def.fallback_chain:
            fb_def = self.registry.get_model(fb_name)
            if not fb_def: continue
            
            fb_node = self._find_node_for_model(fb_def)
            if fb_node:
                return RouteResult(
                    success=True,
                    selected_node=fb_node,
                    selected_model=fb_name,
                    api_url=fb_node.base_url,
                    reasoning=f"Fallback from {req.model_name} to {fb_name} on {fb_node.hostname}"
                )

        return RouteResult(
            success=False, 
            selected_node=None, 
            selected_model=req.model_name, 
            api_url=None, 
            reasoning="All nodes/fallbacks exhausted."
        )

    def _find_node_for_model(self, model: ModelDefinition) -> Optional[NodeCapability]:
        """
        Finds a capable, online node for a specific model definition.
        """
        candidates = []
        
        # Filter by capability
        for node in self.nodes.values():
            if not node.is_online: continue
            if model.model_name not in node.supported_models: continue
            if node.current_load > 0.9: continue # Skip overloaded
            
            # Check preference
            score = 0
            if model.best_node == node.hostname:
                score += 10
            elif model.best_node == "any":
                score += 5
            
            # Prefer GPU for large models
            if model.size_estimate == "large" and node.has_gpu:
                score += 20
            
            candidates.append((score, node))
            
        if not candidates:
            return None
            
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # --- Specific Route Helpers ---

    def route_generate(self, prompt: str, model_name: str) -> RouteResult:
        return self.route_request(RouteRequest(model_name=model_name, task_type="generate"))

    def route_embed(self, text: str, model_name: str) -> RouteResult:
        return self.route_request(RouteRequest(model_name=model_name, task_type="embed"))
    
    def route_classify(self, text: str, model_name: str) -> RouteResult:
        return self.route_request(RouteRequest(model_name=model_name, task_type="classify"))

# --- Demo ---

if __name__ == "__main__":
    registry = ModelRegistry() # Loads defaults
    router = ModelRouter(registry)
    
    print("--- Model Routing Simulation ---\n")
    
    # 1. Happy Path: Heavy Math on Thor
    print("1. Request: Qwen Math (Medium)")
    res = router.route_generate("Solve itergral...", "qwen2.5-math:7b")
    print(f"   -> Success: {res.success}")
    print(f"   -> Node: {res.selected_node.hostname if res.selected_node else 'None'}")
    print(f"   -> Model: {res.selected_model}")
    print(f"   -> Reasoning: {res.reasoning}\n")
    
    # 2. Happy Path: Large Reasoning
    print("2. Request: Llama3 70B (Large)")
    res = router.route_generate("Analyze dataset...", "llama3:70b")
    print(f"   -> Node: {res.selected_node.hostname}\n")
    
    # 3. Failure Scenario: Thor Goes Offline
    print("3. Simulation: Thor crashes!")
    router.update_node_status("thor", False)
    
    print("   Retry: Llama3 70B (Large)")
    res = router.route_generate("Analyze dataset...", "llama3:70b")
    print(f"   -> Success: {res.success}")
    print(f"   -> Node: {res.selected_node.hostname}")
    print(f"   -> Model: {res.selected_model}")
    print(f"   -> Reasoning: {res.reasoning}")
    print("   (Correctly downgraded to 8B on Odin)\n")
    
    # 4. Edge Case: Odin Overloaded
    print("4. Simulation: Odin Load > 90%")
    router.update_node_status("odin", True, load=0.95)
    
    print("   Retry: Phi3 (Small)")
    res = router.route_generate("Status check", "phi3:3.8b")
    print(f"   -> Node: {res.selected_node.hostname}")
    print(f"   -> Reasoning: {res.reasoning}")
    print("   (Routed to Loki as fallback for Odin)\n")
