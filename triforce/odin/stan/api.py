from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import logging

# Imports from our Upgrade (Brains)
from triforce.odin.stan.brains import ParsingBrain, PlanningBrain
from triforce.odin.stan.scheduler import Scheduler
from triforce.odin.stan.execution import ExecutionController
from triforce.odin.stan.awareness import AwarenessBrain
from triforce.odin.stan.registry import WorkerRegistry
from triforce.odin.stan.persona import PersonaBrain

# --- Configuration ---
API_TOKEN_SECRET = "supersecret" # Env var in prod

# --- Data Models ---
class TaskSubmission(BaseModel):
    command: str
    priority: int = 0
    constraints: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    request_id: str
    original_command: str
    status: str
    message: str 
    plan: Optional[List[Any]] = None

class WorkerRegistration(BaseModel):
    node_id: str
    worker_name: str
    url: str
    specs: Dict[str, Any]
    capabilities: List[str]

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.limit = calls_per_minute
        self.clients = {}

    async def check(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        if client_ip in self.clients:
            self.clients[client_ip] = [t for t in self.clients[client_ip] if t > now - 60]
        history = self.clients.get(client_ip, [])
        if len(history) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        history.append(now)
        self.clients[client_ip] = history

limiter = RateLimiter()
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- App Factory ---

def create_app(
    registry: WorkerRegistry,
    parser: ParsingBrain, # Updated Type
    scheduler: Scheduler,
    execution: ExecutionController,
    awareness: AwarenessBrain, # Updated Type
    persona: PersonaBrain,
    planner: Optional[PlanningBrain] = None # New Injection
) -> FastAPI:

    app = FastAPI(title="STAN API", version="2.0.0 (AI)")

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.post("/v1/tasks", response_model=TaskResponse, dependencies=[Depends(limiter.check), Depends(verify_token)])
    async def submit_task(submission: TaskSubmission):
        """
        Submit a natural language command.
        """
        # 1. Parse (AI)
        try:
            graph = await parser.think(submission.command)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Parsing failed: {e}")

        # 2. Plan (AI Optional)
        if planner:
            # Enforce AI constraints (mock)
            await planner.think(graph)

        # 3. Schedule (Algo)
        plan = scheduler.schedule(graph)
        
        if not plan.assignments:
            # Persona explains failure
            msg = await persona.explain_decision("N/A", {"error": "Scheduling failed", "unassigned": plan.unassigned})
            return TaskResponse(
                request_id=graph.request_id,
                original_command=submission.command,
                status="FAILED",
                message=msg.text
            )

        # 4. Dispatch
        exec_payloads = []
        for assign in plan.assignments:
            exec_payloads.append({
                "task_id": assign.task_id,
                "node_id": assign.node_id,
                "worker_url": registry.get_worker(assign.node_id).url,
                "payload": {"code": "AI_JOB_STUB", "args": []}
            })
            
        await execution.dispatch_plan(exec_payloads)

        # 5. Narrate
        narration = await persona.generate_narration("Task Dispatched", {"command": submission.command, "count": len(exec_payloads)})
        return TaskResponse(
            request_id=graph.request_id,
            original_command=submission.command,
            status="ACCEPTED",
            message=narration.text,
            plan=[a.dict() for a in plan.assignments]
        )

    @app.get("/v1/cluster/state", dependencies=[Depends(verify_token)])
    async def get_state():
        """
        Get AI-summarized cluster state.
        """
        state = await awareness.get_cluster_state()
        summary = await persona.summarize_cluster(state.dict())
        return {
            "summary": summary.text,
            "metrics": state.dict()
        }

    # ... Other endpoints (logs, cancel) kept similar or stubbed ...

    return app
