import asyncio
import aiohttp
import time
import logging
import uuid
from typing import Dict, List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field

# --- configuration ---
DEFAULT_TIMEOUT_SEC = 60
MAX_RETRIES = 3
POLL_INTERVAL = 1.0

# --- Data Models ---

class TaskState(BaseModel):
    task_id: str
    node_id: str
    worker_url: str
    status: str = "PENDING" # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    attempts: int = 0
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_error: Optional[str] = None
    result: Optional[dict] = None

class ExecutionEvent(BaseModel):
    event_type: str # STARTED, COMPLETED, FAILED, RETRY
    task_id: str
    timestamp: float = Field(default_factory=time.time)
    details: Optional[dict] = None

# --- Execution Controller ---

class ExecutionController:
    """
    Manages the lifecycle of tasks executing on workers.
    Handles dispatch, polling, timeouts, and retries.
    """

    def __init__(self, event_bus: Optional[Callable[[ExecutionEvent], Awaitable[None]]] = None):
        self.active_tasks: Dict[str, TaskState] = {}
        self.logger = logging.getLogger("stan.execution")
        self.event_bus = event_bus
        self.session = None # Lazy init

    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def dispatch_plan(self, plan_assignments: List[dict]):
        """
        Accepts a list of assignments (from Scheduler) and launches them.
        assignments: [{"task_id": "...", "node": "...", "worker_url": "...", "payload": {...}}]
        """
        tasks = []
        for assign in plan_assignments:
            task_id = assign["task_id"]
            state = TaskState(
                task_id=task_id,
                node_id=assign["node_id"],
                worker_url=assign["worker_url"]
            )
            self.active_tasks[task_id] = state
            
            # Prepare payload
            payload = assign.get("payload", {})
            
            # Schedule execution
            tasks.append(self._execute_with_retry(state, payload))
        
        # Run all initial dispatches in parallel (fire and forget via background tasks usually, 
        # but here we might want to track them. We'll start them as independent tasks.)
        for coro in tasks:
            asyncio.create_task(coro)

    async def _execute_with_retry(self, state: TaskState, payload: dict):
        state.status = "RUNNING"
        state.started_at = time.time()
        await self._emit("STARTED", state.task_id, {"node": state.node_id})

        while state.attempts < MAX_RETRIES:
            state.attempts += 1
            try:
                success = await self._dispatch_http(state, payload)
                if success:
                    # Enter Polling Loop
                    await self._poll_until_completion(state)
                    return
                else:
                    # Dispatch failed immediatley
                    raise Exception("Dispatch rejected by worker")

            except asyncio.CancelledError:
                state.status = "CANCELLED"
                self.logger.info(f"Task {state.task_id} cancelled.")
                return

            except Exception as e:
                state.last_error = str(e)
                self.logger.warning(f"Task {state.task_id} attempt {state.attempts} failed: {e}")
                
                if state.attempts < MAX_RETRIES:
                    await self._emit("RETRY", state.task_id, {"attempt": state.attempts, "error": str(e)})
                    await asyncio.sleep(2 * state.attempts) # Exponential backoff
                else:
                    await self._handle_failure(state, f"Max retries exceeded. Last error: {e}")
                    return

    async def _dispatch_http(self, state: TaskState, payload: dict) -> bool:
        session = await self._get_session()
        url = f"{state.worker_url}/submit" # Assumes standard Thor API
        
        # Standardize STAN payload wrapper if needed
        data = {
            "job_id": state.task_id,
            **payload
        }

        try:
            async with session.post(url, json=data, timeout=10) as resp:
                if resp.status == 200:
                    return True
                else:
                    text = await resp.text()
                    raise Exception(f"Worker returned {resp.status}: {text}")
        except Exception as e:
            self.logger.error(f"Dispatch error to {url}: {e}")
            raise e

    async def _poll_until_completion(self, state: TaskState):
        session = await self._get_session()
        url = f"{state.worker_url}/jobs/{state.task_id}"
        
        start_time = time.time()
        
        while True:
            # Check Timeout
            if (time.time() - start_time) > DEFAULT_TIMEOUT_SEC:
                raise TimeoutError("Task execution timed out")

            try:
                # Poll
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        remote_status = data.get("status")
                        
                        if remote_status == "COMPLETED":
                            state.status = "COMPLETED"
                            state.completed_at = time.time()
                            state.result = data.get("result")
                            await self._on_task_complete(state)
                            return
                        
                        elif remote_status == "FAILED":
                            error_msg = data.get("error") or "Unknown worker error"
                            raise Exception(error_msg)
                            
                        # If RUNNING/PENDING, continue polling
                    
                    elif resp.status == 404:
                         # Worker lost the job?
                         raise Exception("Job not found on worker (404)")
            
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Transient poll error? 
                # If persistent, the retry loop will catch it. 
                # For polling, we might want to be more robust than aborting whole task.
                self.logger.warning(f"Poll error for {state.task_id}: {e}")
                # Re-raising triggers retry of the whole dispatch flow, which might be wrong if job IS running.
                # Ideally check if job exists. For this simple controller, we re-raise.
                raise e

            await asyncio.sleep(POLL_INTERVAL)

    # --- Lifecycle Hooks ---

    async def _on_task_complete(self, state: TaskState):
        self.logger.info(f"Task {state.task_id} COMPLETED on {state.node_id}")
        await self._emit("COMPLETED", state.task_id, {"result": state.result})
        # Clean up if needed, or keep in active_tasks for history

    async def _handle_failure(self, state: TaskState, reason: str):
        state.status = "FAILED"
        state.completed_at = time.time()
        state.last_error = reason
        self.logger.error(f"Task {state.task_id} FAILED: {reason}")
        await self._emit("FAILED", state.task_id, {"error": reason})

    async def _emit(self, event_type: str, task_id: str, details: dict):
        if self.event_bus:
            event = ExecutionEvent(event_type=event_type, task_id=task_id, details=details)
            await self.event_bus(event)

    # --- Management API ---

    async def cancel_task(self, task_id: str):
        if task_id in self.active_tasks:
            state = self.active_tasks[task_id]
            # In a real system, we'd send a cancel signal to the worker too.
            # Here we just stop tracking/polling it if we hold a handle.
            # Since we used create_task and didn't store the Task object, we can't strictly cancel the asyncio task easily 
            # without a map.
            # Simplified: Mark as cancelled, next poll loop will see it? No, poll loop is local.
            # We need to map task_id -> asyncio.Task
            self.logger.info(f"Requested cancellation of {task_id} (Implementing worker-side cancel is TODO)")
            state.status = "CANCELLED"
            await self._emit("CANCELLED", task_id, {})

    async def restart_task(self, task_id: str):
        if task_id in self.active_tasks:
            state = self.active_tasks[task_id]
            if state.status in ["COMPLETED", "FAILED", "CANCELLED"]:
                self.logger.info(f"Restarting {task_id}")
                # Reset state logic...
                state.status = "PENDING"
                state.attempts = 0
                state.last_error = None
                # Re-submit... (Requires keeping the original payload, which current State doesn't fully have)
                # TODO: Store payload in TaskState for restarts
            else:
                self.logger.warning(f"Cannot restart active task {task_id}")

    async def close(self):
        if self.session:
            await self.session.close()
