import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional

# Component Imports
from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig, DEFAULT_OLLAMA_CONFIG, DEFAULT_MOCK_CONFIG
from triforce.odin.stan.model_switcher import ModelSwitcher
from triforce.odin.stan.memory import MemoryEngine, MemoryType
from triforce.odin.stan.brains import ParsingBrain, PlanningBrain, ReflectionBrain
from triforce.odin.stan.awareness import AwarenessBrain
from triforce.odin.stan.persona import PersonaBrain
from triforce.odin.stan.parser import TaskGraph # Data Model

class STANAI:
    """
    The Unified Intelligence Layer of STAN.
    Integrates the 3-Brain Architecture, RAG Memory, Neural Awareness, and Persona.
    
    This class is responsible for the "Thinking" part of the cluster.
    """

    def __init__(self, persistence_path: str = "stan_memory.db", use_mock: bool = True):
        self.logger = logging.getLogger("stan.ai_core")
        
        # 1. Foundation
        config = DEFAULT_MOCK_CONFIG if use_mock else DEFAULT_OLLAMA_CONFIG
        self.ai = AIProviderFactory.create(config)
        self.memory = MemoryEngine(self.ai, db_path=persistence_path)
        
        # 2. Control Systems
        # Mocking registry for AwarenessBrain standalone initialization if needed, 
        # but ideally this is injected. For this core logic wrapper, we might pass it in process.
        # However, AwarenessBrain requires registry at Init. 
        # We will allow late-binding or external injection if needed, but here we instantiate simple placeholders 
        # to ensure the class structure works self-contained for the "AI" part.
        pass # Real init happens in main composition or we mock registry here.
        
        # For awareness, we assume external injection or lazy load. 
        # But to satisfy the prompt's "Output full code" for *this* integration, 
        # we will accept the other components as args or build them if possible.
        # Let's assume we build the "Brain" parts here. 
        
        # We need a Dummy Registry for AwarenessBrain if we want to instantiate it *here* without the full cluster.
        # Let's Stub it.
        self._registry_stub = type("RegistryStub", (), {"list_all": lambda *a: [], "get_worker": lambda *a: None})()
        
        self.awareness = AwarenessBrain(self._registry_stub, self.ai, self.memory)
        self.switcher = ModelSwitcher(config, self.awareness, self.memory)
        
        # 3. The Brains
        self.parser_brain = ParsingBrain(self.ai)
        self.planning_brain = PlanningBrain(self.ai)
        self.reflection_brain = ReflectionBrain(self.ai)
        self.persona_brain = PersonaBrain(self.ai, self.switcher)

    # --- Core Reasoning Methods ---

    async def retrieve_context(self, query: str, limit: int = 3) -> str:
        """
        Fetches RAG context for any cognitive task.
        """
        return await self.memory.get_context_for_reasoning(query)

    async def choose_model(self, task_type: str) -> str:
        """
        Asks the Switcher for the best model.
        """
        return await self.switcher.select_model(task_type)

    async def process_command(self, text: str) -> TaskGraph:
        """
        Stage 1: Input -> Graph.
        """
        context = await self.retrieve_context(text)
        self.logger.info(f"Parsing '{text}' with context size {len(context)} chars")
        
        # Currently ParsingBrain doesn't accept external context in `think`, 
        # but in a real system we'd inject it into the prompt.
        # For now, we assume ParsingBrain handles the prompt construction or we modify it.
        # Let's assume standard behavior.
        return await self.parser_brain.think(text)

    async def generate_plan(self, graph: TaskGraph) -> Dict[str, Any]:
        """
        Stage 2: Graph -> Execution Plan.
        """
        self.logger.info(f"Planning for Request {graph.request_id}")
        return await self.planning_brain.think(graph)

    async def reflect(self, task_id: str, result: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Stage 3: Result -> Learning.
        """
        report = await self.reflection_brain.think(task_id, result, metrics)
        
        # Auto-ingest reflection
        await self.memory.ingest_log({
            "level": "INFO", 
            "message": f"Reflection on {task_id}: {json.dumps(report.dict())}"
        })
        return report

    async def narrate(self, event: str, data: Dict[str, Any]):
        """
        Persona output.
        """
        narrative = await self.persona_brain.generate_narration(event, data)
        return narrative.text

    async def full_reasoning_cycle(self, text: str, execute_callback=None):
        """
        Demonstrates the full pipeline:
        Command -> RAG -> Switcher -> Parser -> Planner -> (Execute) -> Reflection -> Memory.
        """
        start_time = time.time()
        print(f"\n--- [1] Input: '{text}' ---")
        
        # 1. RAG Context
        ctx = await self.retrieve_context(text)
        print(f"--- [2] RAG Context Found: {len(ctx)} chars ---")
        
        # 2. Model Selection (for parsing)
        model = await self.choose_model("parsing")
        print(f"--- [3] Model Selected: {model} ---")
        
        # 3. Parsing
        graph = await self.process_command(text)
        print(f"--- [4] Task Graph Generated: {len(graph.tasks)} tasks ---")
        
        # 4. Planning
        plan_constraints = await self.generate_plan(graph)
        print(f"--- [5] Plan Constraints: {json.dumps(plan_constraints)} ---")
        
        # 5. Execution (Simulated)
        print(f"--- [6] Execution (Simulated) ... ---")
        result = {"status": "success", "output": "Model trained in 45m"} 
        metrics = {"duration_sec": 2700, "gpu_util": 98.5}
        
        # 6. Reflection
        reflection = await self.reflect(graph.tasks[0].task_id, result, metrics)
        print(f"--- [7] Reflection: Success={reflection.success}, Score={reflection.efficiency_score} ---")
        
        # 7. Narration
        narration = await self.narrate("Task Complete", {"task": text, "efficiency": reflection.efficiency_score})
        print(f"--- [8] STAN Speaks: \"{narration}\" ---")
        
        print(f"--- Cycle Complete ({time.time() - start_time:.2f}s) ---\n")

# --- Example Driver ---

async def run_ai_core_demo():
    stan_ai = STANAI(persistence_path=":memory:", use_mock=True)
    
    # Seed Memory
    await stan_ai.memory.add_memory(MemoryType.KNOWLEDGE, "Loki is prone to overheating.", {"source": "manual"})
    
    # Run Cycle
    await stan_ai.full_reasoning_cycle("Train a 70B model on Loki")

if __name__ == "__main__":
    asyncio.run(run_ai_core_demo())
