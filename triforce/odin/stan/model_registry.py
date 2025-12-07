import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# --- Data Models ---

class ModelDefinition(BaseModel):
    model_name: str
    size_estimate: str # e.g. "small", "medium", "large"
    best_node: str = "any" # "odin", "thor", "loki", "local"
    fallback_chain: List[str] = []
    role_tags: List[str] = [] # "parsing", "planning", "persona", "math", "coding"
    performance_score: float = 0.0 # 0.0 - 10.0 based on internal benchmarks
    param_count_b: float = 0.0 # Billions of parameters

# --- Registry ---

class ModelRegistry:
    """
    Central catalog of available models and their capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("stan.registry")
        self._catalog: Dict[str, ModelDefinition] = {}
        self._load_defaults()

    def _load_defaults(self):
        """
        Populate with standard Triforce model set.
        """
        defaults = [
            # Small / Efficient
            ModelDefinition(
                model_name="phi3:3.8b",
                size_estimate="small",
                best_node="local", # Can run on edge
                fallback_chain=[],
                role_tags=["parsing", "json", "fast"],
                performance_score=7.5,
                param_count_b=3.8
            ),
            # Medium / Workhorse
            ModelDefinition(
                model_name="llama3:8b",
                size_estimate="medium",
                best_node="odin",
                fallback_chain=["phi3:3.8b"],
                role_tags=["planning", "general", "persona"],
                performance_score=8.8,
                param_count_b=8.0
            ),
            # Large / Reasoning
            ModelDefinition(
                model_name="llama3:70b",
                size_estimate="large",
                best_node="thor", # GPU heavy
                fallback_chain=["llama3:8b"],
                role_tags=["coding", "complex_reasoning", "analysis"],
                performance_score=9.5,
                param_count_b=70.0
            ),
            # Specialized: Math
            ModelDefinition(
                model_name="qwen2.5-math:7b",
                size_estimate="medium",
                best_node="odin",
                fallback_chain=["llama3:8b"],
                role_tags=["math", "calculation"],
                performance_score=9.2,
                param_count_b=7.0
            ),
             # Specialized: Persona
            ModelDefinition(
                model_name="mistral:7b",
                size_estimate="medium",
                best_node="loki",
                fallback_chain=["llama3:8b"],
                role_tags=["persona", "creative"],
                performance_score=8.5,
                param_count_b=7.0
            ),
            # Specialized: Embedding
            ModelDefinition(
                model_name="nomic-embed-text",
                size_estimate="small",
                best_node="any",
                fallback_chain=[],
                role_tags=["embedding"],
                performance_score=9.0,
                param_count_b=0.1
            )
        ]
        
        for m in defaults:
            self._catalog[m.model_name] = m

    def get_model(self, name: str) -> Optional[ModelDefinition]:
        return self._catalog.get(name)

    def get_model_for_role(self, role: str) -> Optional[ModelDefinition]:
        """
        Finds the best model for a specific role tag.
        Prioritizes performance score.
        """
        candidates = [m for m in self._catalog.values() if role in m.role_tags]
        if not candidates:
            self.logger.warning(f"No model found for role '{role}'. Returning default.")
            return self.get_model("llama3:8b") # Safe default
        
        # Sort by score descending
        candidates.sort(key=lambda x: x.performance_score, reverse=True)
        return candidates[0]

    def get_fallback_models(self, model_name: str) -> List[ModelDefinition]:
        """
        Returns a list of ModelDefinitions for fallbacks.
        """
        primary = self.get_model(model_name)
        if not primary:
            return []
        
        fallbacks = []
        for name in primary.fallback_chain:
            fb = self.get_model(name)
            if fb:
                fallbacks.append(fb)
        return fallbacks

    def list_models(self) -> List[ModelDefinition]:
        return list(self._catalog.values())

# --- Demo / Test ---

if __name__ == "__main__":
    registry = ModelRegistry()
    print("--- Model Registry ---")
    
    print(f"Total Models: {len(registry.list_models())}")
    
    roles = ["math", "planning", "coding", "persona"]
    for r in roles:
        model = registry.get_model_for_role(r)
        print(f"Role '{r}' -> {model.model_name} (Score: {model.performance_score})")

    print("\nFallback Chain for Llama3:70b:")
    fallbacks = registry.get_fallback_models("llama3:70b")
    for f in fallbacks:
        print(f"  -> {f.model_name}")
