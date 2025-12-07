from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import logging

# Imports from our STAN ecosystem
# Assuming these exist based on previous steps
from triforce.odin.stan.parser import CommandParser
from triforce.odin.stan.scheduler import Scheduler, TaskGraph
from triforce.odin.stan.execution import ExecutionController
from triforce.odin.stan.awareness import AwarenessSystem
from triforce.odin.stan.registry import WorkerRegistry, WorkerManifest
from triforce.odin.stan.recovery import RecoveryManager
from triforce.odin.stan.persona import STANPersona

# --- Configuration ---
API_TOKEN_SECRET = "supersecret"  # In prod, load from env

# --- Data Models for API ---

class TaskSubmission(BaseModel):
    command: str
    priority: int = 0
    constraints: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    request_id: str
    original_command: str
    status: str
    message: str # Persona commentary

class WorkerRegistration(BaseModel):
    node_id: str
    worker_name: str
    url: str
    specs: Dict[str, Any]
    capabilities: List[str]

# --- Rate Limiting Stub ---

class RateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.limit = calls_per_minute
        self.clients = {}

    async def check(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        
        # Cleanup old
        if client_ip in self.clients:
            self.clients[client_ip] = [t for t in self.clients[client_ip] if t > now - 60]
            
        history = self.clients.get(client_ip, [])
        if len(history) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        history.append(now)
        self.clients[client_ip] = history

limiter = RateLimiter()

# --- Auth Stub ---
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Application Factory ---

def create_app(
    registry: WorkerRegistry,
    parser: CommandParser,
    scheduler: Scheduler,
    execution: ExecutionController,
    awareness: AwarenessSystem,
    persona: STANPersona
) -> FastAPI:

    app = FastAPI(
        title="STAN API",
        description="External interface for the Triforce Cluster Orchestrator",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Endpoints ---

    @app.post("/v1/tasks", response_model=TaskResponse, dependencies=[Depends(limiter.check), Depends(verify_token)])
    async def submit_task(submission: TaskSubmission):
        """
        Submit a natural language command to the cluster.
        """
        # 1. Parse
        try:
            graph = parser.parse(submission.command)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Parsing failed: {e}")

        # 2. Schedule
        plan = scheduler.schedule(graph)
        
        if not plan.assignments and plan.unassigned:
             # Persona explains why we failed
             msg = persona.speak("scheduling_rationale", {"task_id": "REQ-FAIL", "node": "None"}).text
             return TaskResponse(
                 request_id=graph.request_id,
                 original_command=submission.command,
                 status="FAILED",
                 message=f"Scheduling failed: {msg}"
             )

        # 3. Execute
        # Convert assignments to generic dicts for controller
        exec_payloads = []
        for assign in plan.assignments:
            exec_payloads.append({
                "task_id": assign.task_id,
                "node_id": assign.node_id,
                "worker_url": registry.get_worker(assign.node_id).url,
                "payload": {"code": "TODO_INJECT_CODE", "args": []} # In real impl, TaskNode has code
            })
            
        await execution.dispatch_plan(exec_payloads)

        # 4. Respond
        msg = persona.speak("current_activity").text
        return TaskResponse(
            request_id=graph.request_id,
            original_command=submission.command,
            status="ACCEPTED",
            message=f"Tasks dispatched. {msg}"
        )

    @app.get("/v1/cluster/state", dependencies=[Depends(verify_token)])
    async def get_state():
        """
        Get high-level cluster metrics and health.
        """
        state = awareness.get_cluster_state()
        # Wrap in persona response
        p_resp = persona.speak("cluster_state")
        return {
            "summary": p_resp.text,
            "metrics": state.dict()
        }

    @app.get("/v1/logs", dependencies=[Depends(verify_token)])
    async def get_logs(limit: int = 50):
        """
        Fetch execution logs (Stubbed).
        """
        # Real impl would query database/log aggregator
        return {"logs": ["Log entry 1", "Log entry 2"]}

    @app.post("/v1/tasks/{task_id}/cancel", dependencies=[Depends(verify_token)])
    async def cancel_task(task_id: str):
        await execution.cancel_task(task_id)
        return {"status": "cancelled", "task_id": task_id}

    @app.post("/v1/workers", dependencies=[Depends(verify_token)])
    async def register_worker(worker: WorkerRegistration):
        """
        Manual registration (Auto-discovery usually handles this).
        """
        try:
            registry.add_worker(worker.dict())
            return {"status": "registered", "node_id": worker.node_id}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/v1/workers/{node_id}", dependencies=[Depends(verify_token)])
    async def unregister_worker(node_id: str):
        registry.remove_worker(node_id)
        return {"status": "removed", "node_id": node_id}

    return app

# --- Standalone Entrypoint (for testing) ---
if __name__ == "__main__":
    import uvicorn
    # Mocks for standalone run
    class MockObj: 
        def __getattr__(self, name): return lambda *a, **k: MockObj()
        def parse(self, c): return MockObj()
        def schedule(self, g): return MockObj()
        def speak(self, i, c=None): return MockObj()
    
    # In real usage, main.py would inject real objects
    app = create_app(MockObj(), MockObj(), MockObj(), MockObj(), MockObj(), MockObj())
    uvicorn.run(app, host="0.0.0.0", port=9000)
