import uvicorn
import asyncio
import os
import uuid
import time
import threading
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from triforce.thor.utils.logger import logger, log_buffer
from triforce.thor.metrics.collector import init_gpu, get_system_metrics
from triforce.thor.sandbox.executor import execute_python_code
from triforce.thor.worker.core import heartbeat_loop
from triforce.common.models.jobs import JobRequest
from triforce.thor.sandbox.executor import JobResult

NODE_ID = str(uuid.uuid4())
START_TIME = time.time()
PORT = int(os.getenv("PORT", 8000))
API_TOKEN = os.getenv("API_TOKEN", "default-insecure-token")

security = HTTPBearer()

async def verify_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    
    role = request.headers.get("X-Role", "unknown")
    logger.info(f"Authenticated request from role: {role}", extra={"event": "auth", "role": role})

app = FastAPI(title="THOR Worker Node", dependencies=[Depends(verify_token)])

# Global Job State
active_jobs_count = 0
active_jobs_lock = threading.Lock()

@app.on_event("startup")
async def startup_event():
    # Start heartbeat
    t = threading.Thread(target=heartbeat_loop, args=(NODE_ID, PORT, API_TOKEN), name="HeartbeatThread", daemon=True)
    t.start()
    
    # Init GPU
    init_gpu()
    logger.info(f"THOR Online: {NODE_ID}", extra={"event": "startup", "role": "worker"})

@app.get("/metrics")
def get_metrics():
    with active_jobs_lock:
        count = active_jobs_count
    return get_system_metrics(START_TIME, NODE_ID, count)

@app.get("/logs")
def get_logs(limit: int = 100):
    all_logs = list(log_buffer.formatted_buffer)
    return all_logs[-limit:]

@app.get("/health")
def health_check():
    return {"status": "ok", "node_id": NODE_ID}

@app.post("/work", response_model=JobResult)
async def submit_work(job: JobRequest):
    global active_jobs_count
    
    with active_jobs_lock:
        active_jobs_count += 1
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, execute_python_code, job)
        return result
    finally:
        with active_jobs_lock:
            active_jobs_count -= 1

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
