import uvicorn
import asyncio
import os
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from triforce.odin.controllers.worker import ClusterManager
from triforce.odin.scheduler.core import Scheduler
from triforce.odin.api import routes
from triforce.odin.utils.logger import logger
from triforce.common.storage.client import StorageClient
from triforce.odin.workflows.engine import WorkflowManager

PORT = int(os.getenv("PORT", 8080))
API_TOKEN = os.getenv("API_TOKEN", "default-insecure-token")
security = HTTPBearer(auto_error=False)

async def verify_token(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    print(f"DEBUG: Entering verify_token {request.url.path}", flush=True)
    token = None
    
    # 1. Check Bearer Header
    if credentials:
        token = credentials.credentials
    
    # 2. Check Query Parameter (for browser dashboard)
    if not token:
        token = request.query_params.get("token")
        
    if token != API_TOKEN:
        # Avoid logging annoying checks for health/metrics if public, but for now strict
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    
    role = request.headers.get("X-Role", "unknown")
    if role != "unknown":
        logger.info(f"Authenticated request from role: {role}", extra={"event": "auth", "role": role})

app = FastAPI(title="ODIN Controller (Triforce)", dependencies=[Depends(verify_token)])

from triforce.odin.stan.supernova import STANSupernova

# Initialize components
cluster = ClusterManager()
scheduler = Scheduler(cluster)
workflow_manager = WorkflowManager(scheduler, scheduler.store)
# Check if we have a key for Gemini, otherwise mock
use_mock = os.getenv("GEMINI_API_KEY") is None
stan = STANSupernova(use_mock=use_mock) # Unified AI Core

# Inject dependencies into router (Naive injection, or better using Depends)
routes.cluster = cluster
routes.scheduler = scheduler
routes.storage = StorageClient() # MinIO connection
routes.workflow_manager = workflow_manager
routes.stan = stan

app.include_router(routes.router)

@app.on_event("startup")
async def startup():
    logger.info("ODIN Controller Online", extra={"event": "startup", "role": "controller"})
    asyncio.create_task(cluster.update_health())
    asyncio.create_task(scheduler.run())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
