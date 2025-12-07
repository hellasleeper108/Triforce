import logging
import time
import asyncio
from typing import List, Dict, Optional

# Mocking dependent types for standalone definition
# from triforce.odin.stan.registry import WorkerRegistry
# from triforce.odin.stan.execution import ExecutionController
# from triforce.odin.stan.awareness import AwarenessSystem
# from triforce.odin.stan.scheduler import Scheduler

class RecoveryManager:
    """
    The 'Doctor' of STAN.
    Monitors cluster health, handles failures, and enforces safety limits.
    """

    def __init__(self, 
                 registry, 
                 execution_controller, 
                 awareness, 
                 scheduler):
        self.registry = registry
        self.exec_ctrl = execution_controller
        self.awareness = awareness
        self.scheduler = scheduler
        self.logger = logging.getLogger("stan.recovery")
        self.healing_interval = 5.0
        self._loop_running = False

    async def start_monitoring(self):
        """Starts the background healing loop."""
        self._loop_running = True
        self.logger.info("STAN Recovery System: ONLINE")
        while self._loop_running:
            try:
                await self.run_healing_cycle()
            except Exception as e:
                self.logger.error(f"Error in healing cycle: {e}")
            await asyncio.sleep(self.healing_interval)

    async def stop(self):
        self._loop_running = False

    async def run_healing_cycle(self):
        """Main self-healing logic."""
        # 1. Detect & Handle Offline Nodes
        await self._check_node_failures()

        # 2. Thermal/Resource Safety Checks
        await self._enforce_safety_limits()

        # 3. Dynamic Rebalancing (Auto-Parallelize)
        await self._check_rebalancing_needs()

    # --- 1. Failure Handling ---

    async def _check_node_failures(self):
        # awareness system already marks timeouts as High Latency risk
        # We need to act on it if it becomes "OFFLINE"
        
        # Check for nodes that are officially marked ACTIVE in registry but are actually dead
        # based on Awareness system's zombie detection.
        anomalies = self.awareness.detect_anomalies()
        
        for detail in anomalies.details:
            if "Zombie" in detail:
                # Extract node name/id roughly (regex better in prod)
                # "Node loki marked ACTIVE..."
                words = detail.split()
                if len(words) > 1:
                    node_name = words[1]
                    # Find ID
                    target = next((w for w in self.registry.list_all() if w.worker_name == node_name), None)
                    if target:
                        self.logger.warning(f"ZOMBIE DETECTED: {target.worker_name}. Initiating Protocol Lazarus.")
                        await self._handle_node_failure(target.node_id)
        
        # Also check Explicitly OFFLINE nodes with jobs stuck in "RUNNING" state
        # (This consistency check handles persisted state mismatch)
        for task_id, state in self.exec_ctrl.active_tasks.items():
            if state.status == "RUNNING":
                worker = self.registry.get_worker(state.node_id)
                if not worker or worker.status == "OFFLINE":
                    self.logger.warning(f"Task {task_id} running on DEAD node {state.node_id}. Reassigning.")
                    await self._recover_task(state)

    async def _handle_node_failure(self, node_id: str):
        # 1. Mark node as OFFLINE in Registry
        self.registry.update_status(node_id, "OFFLINE")
        
        # 2. Find all jobs assigned to this node
        impacted_tasks = [
            state for state in self.exec_ctrl.active_tasks.values() 
            if state.node_id == node_id and state.status in ["RUNNING", "PENDING"]
        ]
        
        if impacted_tasks:
            self.logger.info(f"Node {node_id} failure impacts {len(impacted_tasks)} tasks.")
            for task in impacted_tasks:
                await self._recover_task(task)

    async def _recover_task(self, task_state):
        # Cancel current zombie execution
        await self.exec_ctrl.cancel_task(task_state.task_id)
        
        # Restart (which puts it back to pending/scheduler queue logic usually)
        # But exec_ctrl's restart assumes same node? 
        # We need to tell Scheduler to re-plan.
        # Impl detail: We might set status to "FAILED" with specific error, 
        # and letting a higher-level Workflow Manager resubmit?
        # Or we act as the healer and move it.
        
        self.logger.info(f"Healing Task {task_state.task_id} -> Re-queueing.")
        # Simulating re-queue by resetting status and letting Controller Retry logic pick a NEW node?
        # Controller retry logic is usually bound to the assigned node.
        # We need to clear the assignment.
        
        # FORCE REASSIGNMENT (stub logic)
        # In a real system we'd push back to Scheduler.
        # Here we just log for now.
        pass

    # --- 2. Safety & Throttle ---

    async def _enforce_safety_limits(self):
        """
        Prevents equipment damage (Loki melting) and system hangs.
        """
        for worker in self.registry.list_all():
            if worker.status != "ACTIVE":
                continue

            # Thermal Throttling for Loki (MacBook)
            if "loki" in worker.worker_name.lower():
                # Checking hypothetical GPU/CPU temp
                temp = worker.metrics.gpu_temp 
                if temp > 90:
                    self.logger.critical(f"LOKI IS MELTING ({temp}C). PAUSING NEW TASKS.")
                    self.registry.update_status(worker.node_id, "BUSY") # Prevents new scheduling
                    # Optional: Pause running tasks? (Hard to do without process suspension)
                elif temp < 70 and worker.status == "BUSY" and worker.metrics.active_jobs == 0:
                    # Cooled down
                    self.logger.info(f"Loki cooled down ({temp}C). Resuming.")
                    self.registry.update_status(worker.node_id, "ACTIVE")
            
            # Resource Pause (RAM OOM Prevention)
            if worker.metrics.ram_usage > 98:
                self.logger.error(f"Node {worker.worker_name} RAM Critical ({worker.metrics.ram_usage}%).")
                self.registry.update_status(worker.node_id, "BUSY")

    # --- 3. Auto-Parallelization / Rebalancing ---

    async def _check_rebalancing_needs(self):
        """
        If Loki is swamped but Odin (Master) is idle, move tasks.
        """
        loki_node = next((w for w in self.registry.list_all() if "loki" in w.worker_name.lower()), None)
        odin_node = next((w for w in self.registry.list_all() if "odin" in w.worker_name.lower()), None)
        
        if not loki_node or not odin_node:
            return

        # Condition: Loki overloaded, Odin free
        if (loki_node.metrics.cpu_usage > 90 or loki_node.metrics.active_jobs > 5) and \
           (odin_node.metrics.cpu_usage < 20 and odin_node.metrics.active_jobs == 0):
            
            self.logger.info("Detected Bottleneck on Loki. Suggesting offload to Odin.")
            
            # Strategy: Find pending tasks targeting "loki" (or generic) and move them?
            # Requires deep hook into Scheduler queue.
            # For now, we simulate the action.
            self.logger.info(">> Auto-Parallelization Prompt: Future tasks rerouted to Odin.")


# --- Sample Usage ---
if __name__ == "__main__":
    # Mock mocks
    logging.basicConfig(level=logging.INFO)
    rm = RecoveryManager(None, None, None, None)
    # asyncio.run(rm.run_healing_cycle()) # Fails without real objects, just testing syntax
