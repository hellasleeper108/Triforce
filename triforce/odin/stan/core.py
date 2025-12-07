import asyncio
import logging
from typing import Dict, Any, Optional

# Component Imports
# Provider & Switcher
from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig, DEFAULT_OLLAMA_CONFIG, DEFAULT_MOCK_CONFIG
from triforce.odin.stan.model_switcher import ModelSwitcher

# Brains
from triforce.odin.stan.parser import CommandParser # Needs upgrade to ParsingBrain eventually, but keeping compatible for now or assuming brains.py integration
from triforce.odin.stan.brains import ParsingBrain, PlanningBrain, ReflectionBrain
from triforce.odin.stan.persona import PersonaBrain
from triforce.odin.stan.memory import MemoryEngine, MemoryType
from triforce.odin.stan.awareness import AwarenessBrain

# Core Logic
from triforce.odin.stan.registry import WorkerRegistry
from triforce.odin.stan.scheduler import Scheduler, TaskGraph
from triforce.odin.stan.execution import ExecutionController
from triforce.odin.stan.recovery import RecoveryManager
from triforce.odin.stan.api import create_app

class STAN:
    """
    System for Task Automation & Navigation (STAN).
    Master Orchestrator v2 (AI-Enhanced).
    """

    def __init__(self, persistence_path: Optional[str] = None, use_mock_ai: bool = True, ai_provider: Optional[Any] = None, memory_engine: Optional[Any] = None):
        self.logger = logging.getLogger("stan.core")
        
        # 0. AI Layer
        if ai_provider:
            self.ai = ai_provider
            # If explicit provider given, config is assumed handled by it or irrelevant
            # We construct a dummy config for switcher if needed, or switcher needs update
            # For now, we assume provider is fully capable.
            ai_config = DEFAULT_OLLAMA_CONFIG # Fallback config for switcher params
        else:
            ai_config = DEFAULT_MOCK_CONFIG if use_mock_ai else DEFAULT_OLLAMA_CONFIG
            self.ai = AIProviderFactory.create(ai_config)
        
        # 1. Foundation
        self.registry = WorkerRegistry(persistence_path=persistence_path)
        
        if memory_engine:
            self.memory = memory_engine
        else:
            self.memory = MemoryEngine(self.ai, db_path="stan_memory.db")
        
        # 2. Senses & Cortex
        self.awareness = AwarenessBrain(self.registry, self.ai, self.memory)
        # Note: ModelSwitcher might need refactoring to accept provider directly too
        # reusing ai_config for now as legacy param
        self.switcher = ModelSwitcher(ai_config, self.awareness, self.memory)
        
        # 3. Cognition (The Three Brains)
        self.parser_brain = ParsingBrain(self.ai) 
        self.planning_brain = PlanningBrain(self.ai)
        self.reflection_brain = ReflectionBrain(self.ai)
        
        # 4. Old School Scheduler (Deterministic fallback / Base)
        self.scheduler = Scheduler(self.registry)
        
        # 5. Interface
        self.persona = PersonaBrain(self.ai, self.switcher)
        
        # 6. Action
        self.execution = ExecutionController(event_bus=self._handle_event)
        self.recovery = RecoveryManager(
            self.registry, 
            self.execution, 
            self.awareness,
            self.scheduler
        )
        
        self._background_tasks = []

    # --- Lifecycle ---

    async def start(self):
        self.logger.info("Initializing STAN Core (AI Online)...")
        self._background_tasks.append(asyncio.create_task(self.recovery.start_monitoring()))
        self.logger.info("STAN Online.")

    async def stop(self):
        self.logger.info("STAN Shutting down...")
        await self.recovery.stop()
        await self.execution.close()
        for t in self._background_tasks:
            t.cancel()

    # --- Public API Methods ---

    def get_api_app(self):
        return create_app(
            self.registry,
            self.parser_brain, # Pass new brain
            self.scheduler,
            self.execution,
            self.awareness,
            self.persona,
            # Inject new components if API needs them
            self.planning_brain
        )

    async def run(self, command: str) -> Dict[str, Any]:
        """
        End-to-end execution flow.
        """
        self.logger.info(f"Command: '{command}'")
        
        # 1. Parse (AI)
        graph = await self.parser_brain.think(command)
        
        # 2. Plan (AI + Deterministic)
        # AI Planner refines it, but we still use 'Scheduler' for node assignment?
        # Let's say AI Planner refines constraints, Scheduler maps to nodes.
        ai_plan_constraints = await self.planning_brain.think(graph)
        # Apply AI constraints to graph (Mock logic)
        
        # 3. Schedule (Deterministic)
        plan = self.scheduler.schedule(graph)
        
        if not plan.assignments and plan.unassigned:
            explanation = await self.persona.explain_decision("N/A", {"node": "None", "error": "Resources unavailable"})
            return {
                "status": "failed",
                "message": explanation.text,
                "unassigned": plan.unassigned
            }

        # 4. Dispatch
        exec_list = []
        for assign in plan.assignments:
            worker = self.registry.get_worker(assign.node_id)
            payload = {
                "job_type": "compute", 
                "code": "print('STAN AI Job')", 
                "args": [], 
                "entrypoint": "main"
            }
            exec_list.append({
                "task_id": assign.task_id,
                "node_id": assign.node_id,
                "worker_url": worker.url,
                "payload": payload
            })

        await self.execution.dispatch_plan(exec_list)
        
        # 5. Narrate
        narration = await self.persona.generate_narration("Command Accepted", {"command": command, "jobs": len(exec_list)})
        return {
            "status": "success", 
            "request_id": graph.request_id,
            "message": narration.text,
            "tone": narration.tone,
            "plan": [a.dict() for a in plan.assignments]
        }

    async def get_cluster_state_report(self) -> Dict[str, Any]:
        state = await self.awareness.get_cluster_state()
        summary = await self.persona.summarize_cluster(state.dict())
        return {
            "summary": summary.text,
            "metrics": state.dict()
        }

    async def explain(self, task_id: str) -> str:
        state = self.execution.active_tasks.get(task_id)
        if not state:
            return "Task unknown."
        
        resp = await self.persona.explain_decision(task_id, {"node": state.node_id, "status": state.status})
        return resp.text

    # --- Internal Event Bus ---

    async def _handle_event(self, event):
        self.logger.debug(f"Event: {event.event_type}")
        
        # Ingest to Memory
        if event.event_type == "FAILED":
            await self.memory.ingest_log({"level": "ERROR", "message": f"Task {event.task_id} Failed: {event.details}"})
        
        # Trigger Reflection if complete
        if event.event_type == "COMPLETED":
            report_data = {
                "success": True, 
                "efficiency_score": 0.9, # Mock
                "anomalies": []
            }
            reflection = await self.reflection_brain.think(event.task_id, {"status": "ok"}, {})
            
            # Store reflection
            await self.memory.ingest_log({"level": "INFO", "message": f"Reflection on {event.task_id}: {reflection.efficiency_score}"})

# --- Standalone Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    stan = STAN()
    app = stan.get_api_app()
    uvicorn.run(app, host="0.0.0.0", port=9000)
