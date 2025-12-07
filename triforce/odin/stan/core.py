import asyncio
import logging
from typing import Dict, Any, Optional

# Component Imports
from triforce.odin.stan.parser import CommandParser
from triforce.odin.stan.registry import WorkerRegistry
from triforce.odin.stan.scheduler import Scheduler, TaskGraph
from triforce.odin.stan.execution import ExecutionController
from triforce.odin.stan.awareness import AwarenessSystem
from triforce.odin.stan.recovery import RecoveryManager
from triforce.odin.stan.persona import STANPersona
from triforce.odin.stan.api import create_app

class STAN:
    """
    System for Task Automation & Navigation (STAN).
    The Master Orchestrator Object.
    
    Ties together the Brain (Parser/Persona), the Nervous System (Execution/Recovery),
    and the Senses (Awareness/Registry).
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.logger = logging.getLogger("stan.core")
        
        # 1. Foundation (Registry & Senses)
        self.registry = WorkerRegistry(persistence_path=persistence_path)
        self.awareness = AwarenessSystem(self.registry)
        
        # 2. Cognition (Parser & Planner)
        self.parser = CommandParser() # Can inject LLM client here
        self.scheduler = Scheduler(self.registry)
        
        # 3. Action (Controller & Healer)
        self.execution = ExecutionController(event_bus=self._handle_event)
        self.recovery = RecoveryManager(
            self.registry, 
            self.execution, 
            self.awareness,
            self.scheduler
        )
        
        # 4. Interface (Persona)
        self.persona = STANPersona(self.awareness, self.scheduler)
        
        # Internal State
        self._background_tasks = []

    # --- Lifecycle ---

    async def start(self):
        """Ignition."""
        self.logger.info("Initializing STAN Core...")
        
        # Start Recovery Monitor
        self._background_tasks.append(asyncio.create_task(self.recovery.start_monitoring()))
        
        # Start API (If embedded) - Usually API runs separate Uvicorn, 
        # but here we prepare dependencies.
        
        self.logger.info("STAN Online. Awaiting commands.")

    async def stop(self):
        """Shutdown."""
        self.logger.info("STAN Shutting down...")
        await self.recovery.stop()
        await self.execution.close()
        for t in self._background_tasks:
            t.cancel()

    # --- Public API Methods ---

    def get_api_app(self):
        """Returns the FastAPI app configured with this STAN instance."""
        return create_app(
            self.registry,
            self.parser,
            self.scheduler,
            self.execution,
            self.awareness,
            self.persona
        )

    async def run(self, command: str) -> Dict[str, Any]:
        """
        Main Entrypoint: NL Command -> Execution.
        """
        self.logger.info(f"Command received: '{command}'")
        
        # 1. Parse
        try:
            graph = self.parser.parse(command)
        except Exception as e:
            return {"status": "error", "message": f"I couldn't understand that: {e}"}

        # 2. Schedule
        plan = self.scheduler.schedule(graph)
        
        if not plan.assignments and plan.unassigned:
            explanation = self.persona.speak("scheduling_rationale", {"task_id": "REQ-FAIL", "node": "None"}).text
            return {
                "status": "failed",
                "message": f"Scheduling failed. {explanation}",
                "unassigned": plan.unassigned
            }

        # 3. Dispatch
        # Transform plan to execution payloads
        exec_list = []
        for assign in plan.assignments:
            worker = self.registry.get_worker(assign.node_id)
            # Find task detail provided by 'graph' (Need lookup map)
            task_node = next(t for t in graph.tasks if t.task_id == assign.task_id)
            
            # Construct real payload
            payload = {
                "job_type": "compute", # Simplification, parser should provide
                "code": "print('Hello from STAN')", # Placeholder until parser generates real code
                "args": task_node.execution_steps[0].args, # Naive mapping
                "entrypoint": "main"
            }
            
            exec_list.append({
                "task_id": assign.task_id,
                "node_id": assign.node_id,
                "worker_url": worker.url,
                "payload": payload
            })

        await self.execution.dispatch_plan(exec_list)
        
        # 4. Confirmation
        msg = self.persona.speak("current_activity").text
        return {
            "status": "success", 
            "request_id": graph.request_id,
            "message": f"Command accepted. {msg}",
            "plan": [a.dict() for a in plan.assignments]
        }

    def get_cluster_state(self) -> Dict[str, Any]:
        """Returns high-level status wrapped in Persona flavor."""
        state = self.awareness.get_cluster_state()
        commentary = self.persona.speak("cluster_state").text
        return {
            "summary": commentary,
            "metrics": state.dict()
        }

    def explain(self, task_id: str) -> str:
        """Asks STAN to explain a specific task's status/assignment."""
        # Find task state
        state = self.execution.active_tasks.get(task_id)
        if not state:
            return f"I have no record of task {task_id}. Perhaps it was a fever dream?"
            
        return self.persona.speak("scheduling_rationale", {"task_id": task_id, "node": state.node_id}).text

    def full_status_report(self) -> Dict[str, Any]:
        """Comprehensive system diagnostics."""
        return {
            "awareness": self.awareness.get_cluster_state().dict(),
            "active_tasks": len(self.execution.active_tasks),
            "anomalies": self.awareness.detect_anomalies().dict(),
            "nodes": [w.dict() for w in self.registry.list_all()]
        }

    # --- Internal Event Bus ---

    async def _handle_event(self, event):
        """
        Stream events from ExecutionController -> Awareness -> Logs
        """
        # Ingest metrics if provided
        # Update specific node status based on task success/failure?
        self.logger.debug(f"Event: {event.event_type} for {event.task_id}")
        
        if event.event_type == "FAILED":
             self.recovery.logger.warning(f"Task Failure Event: {event.details}")
             # RecoveryManager actively monitors state, but could trigger here too.

# --- Entrypoint for Testing ---
if __name__ == "__main__":
    import uvicorn
    # Standalone run
    stan = STAN()
    app = stan.get_api_app()
    print("STAN Core assembled. Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=9000)
