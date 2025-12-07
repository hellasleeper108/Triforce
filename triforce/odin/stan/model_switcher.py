import logging
import asyncio
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# Component Imports
from triforce.odin.stan.ai_provider import ProviderConfig, ModelConfig
from triforce.odin.stan.awareness import AwarenessBrain
from triforce.odin.stan.memory import MemoryEngine, MemoryType

class SelectionReport(BaseModel):
    selected_model: str
    candidates_considered: int
    reasoning: List[str]
    fallback_path: List[str]

class ModelSwitcher:
    """
    The 'Cortex' of STAN.
    Decides WHICH brain to use for a given thought.
    """

    # Complexity -> Min Size (Billions) mapping
    SIZE_MAP = {
        "tiny": 0.5,   # Telemetry / Classification
        "small": 3.0,  # Parsing / Persona
        "medium": 7.0, # Standard Planning
        "large": 13.0, # Deep Reasoning
        "huge": 70.0   # Complex Coding / Analysis
    }

    # Task Type -> Default Complexity
    TASK_DEFAULTS = {
        "telemetry": "tiny",
        "parsing": "small",
        "persona": "small",
        "planning": "medium",
        "reasoning": "large",
        "coding": "huge"
    }

    def __init__(self, provider_config: ProviderConfig, awareness: AwarenessBrain, memory: MemoryEngine):
        self.config = provider_config
        self.awareness = awareness
        self.memory = memory
        self.logger = logging.getLogger("stan.switcher")

    async def select_model(self, task_type: str, complexity: str = None) -> str:
        """
        Main API. Returns the name of the model to use.
        """
        report = await self._evaluate_options(task_type, complexity)
        self.logger.info(f"Selected {report.selected_model} for {task_type} (Reason: {report.reasoning[0]})")
        return report.selected_model

    async def _evaluate_options(self, task_type: str, complexity: str = None) -> SelectionReport:
        reasoning = []
        
        # 1. Determine Target Complexity
        if not complexity:
            complexity = self.TASK_DEFAULTS.get(task_type, "medium")
        
        target_size = self.SIZE_MAP.get(complexity, 7.0)
        reasoning.append(f"Task '{task_type}' maps to complexity '{complexity}' (Min {target_size}B params)")

        # 2. Fetch Candidates from Config
        candidates = [m for m in self.config.models if "generate" in m.capabilities]
        
        # 3. Check User Preferences (RAG)
        # We search memory for "model preference"
        pref_context = await self.memory.search_memory("preferred model for code", type_filter=MemoryType.USER_PREF, limit=1)
        user_override = None
        if pref_context:
            # Naive extraction - real system would parse more carefully
            text = pref_context[0].item.text.lower()
            if "prefer" in text:
                for m in candidates:
                    if m.name.lower() in text:
                        user_override = m.name
                        reasoning.append(f"User preference found: {user_override}")
                        break

        # 4. Check Cluster Status (Awareness)
        # If cluster is melting, downgrade complexity?
        cluster_state = await self.awareness.get_cluster_state()
        downgrade_mode = False
        
        # Example heuristic: If global load > 80% or warnings exist
        if cluster_state.avg_load > 80 or cluster_state.warnings:
            downgrade_mode = True
            reasoning.append("Cluster under load/stress. Preferring smaller/faster models.")

        # 5. Selection Logic
        best_match = None
        
        # Filter by size
        suitable = [m for m in candidates if m.size_params_b >= target_size]
        if not suitable and candidates:
            # Fallback to largest available if none meet criteria
            suitable = [max(candidates, key=lambda x: x.size_params_b)]
            reasoning.append("No models meet size requirement. Falling back to largest available.")
        
        if downgrade_mode:
            # Pick smallest suitable
            suitable.sort(key=lambda x: x.size_params_b)
        else:
            # Pick smallest suitable that meets threshold (efficiency) 
            # OR typically we want 'just enough'.
            suitable.sort(key=lambda x: x.size_params_b)

        # Apply override if valid
        if user_override and any(m.name == user_override for m in suitable):
             best_match = user_override
        elif suitable:
             best_match = suitable[0].name
        else:
             best_match = self.config.default_model # Absolute fallback
             
        return SelectionReport(
            selected_model=best_match,
            candidates_considered=len(candidates),
            reasoning=reasoning,
            fallback_path=[]
        )

# --- Example Driver ---

async def run_switcher_demo():
    # 1. Mocks
    from triforce.odin.stan.ai_provider import ProviderConfig, ModelConfig
    
    mock_config = ProviderConfig(
        type="mock",
        models=[
            ModelConfig(name="tiny-llama-1b", size_params_b=1.1, capabilities=["generate"], ideal_platform="cpu"),
            ModelConfig(name="mistral-7b", size_params_b=7.0, capabilities=["generate"], ideal_platform="metal"),
            ModelConfig(name="llama3-70b", size_params_b=70.0, capabilities=["generate"], ideal_platform="gpu")
        ],
        default_model="mistral-7b"
    )
    
    class MockAwareness:
        async def get_cluster_state(self):
            return type("S",(),{"avg_load": 10, "warnings": []})()

    class MockMemory:
        async def search_memory(self, q, type_filter, limit):
            if "code" in q:
                return [type("Result",(),{"item": type("Item",(),{"text": "I prefer llama3-70b for coding tasks."})()})()]
            return []

    # 2. Init
    switcher = ModelSwitcher(mock_config, MockAwareness(), MockMemory())
    
    print("=== Model Switching Engine ===\n")
    
    # Scenario A: Telemetry (Tiny)
    model = await switcher.select_model("telemetry")
    print(f"[A] Telemetry Check -> {model}")
    
    # Scenario B: Planning (Medium)
    model = await switcher.select_model("planning")
    print(f"[B] General Planning -> {model}")
    
    # Scenario C: Coding (Huge + User Pref)
    model = await switcher.select_model("coding")
    print(f"[C] Hard Coding Task -> {model}")
    
    # Scenario D: Load Stress (Downgrade)
    class StressedAwareness:
        async def get_cluster_state(self): return type("S",(),{"avg_load": 95, "warnings": ["High Load"]})()
    
    switcher.awareness = StressedAwareness()
    model = await switcher.select_model("reasoning", complexity="large") # Requesting large
    # Logic might still pick 70b if it's the only one > 13b, 
    # but let's see if our logic sorts by size ASC per suitable list.
    print(f"[D] Reasoning under Load -> {model}")

if __name__ == "__main__":
    asyncio.run(run_switcher_demo())
