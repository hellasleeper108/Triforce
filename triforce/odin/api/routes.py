from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uuid
import asyncio
import time
import os
import collections
import json

from triforce.common.models.jobs import JobSubmission, JobRequest, JobResponse
from triforce.odin.controllers.worker import RegistrationData
from triforce.odin.scheduler.core import InternalJob
from triforce.odin.utils.logger import logger, log_buffer
import httpx

router = APIRouter()
templates = Jinja2Templates(directory="triforce/dashboard/templates")

from triforce.common.storage.client import StorageClient

from typing import Dict, Any

# Globals (Injected from main)
cluster = None
scheduler = None
storage = None # Will be injected
workflow_manager = None

@router.post("/register")
def register_worker(data: RegistrationData):
    cluster.register(data)
    return {"status": "registered"}

@router.get("/nodes")
async def get_nodes():
    return cluster.get_snapshot()

@router.get("/cluster/state")
async def get_cluster_state():
    return {
        "timestamp": time.time(),
        "worker_count": len(cluster.workers),
        "topology": cluster.get_snapshot()
    }

async def event_generator(request: Request):
    while True:
        if await request.is_disconnected():
            break
            
        workers = cluster.get_snapshot()
        # Sort: Active first, then by name
        workers.sort(key=lambda w: (w.status != "ACTIVE", w.specs.get("worker_name", "")))
        
        scores = {w.url: cluster._calculate_score(w) for w in workers}
        
        # Serialize workers to dicts because dataclasses aren't automatically JSON serializable in json.dumps
        # We need a custom encoder or helper. But wait, get_snapshot returns Worker objects.
        # Worker objects are dataclasses.
        
        worker_data = []
        for w in workers:
            worker_data.append({
                "url": w.url,
                "node_id": w.node_id,
                "status": w.status,
                "specs": w.specs,
                "metrics": w.metrics,
                "active_jobs": w.active_jobs,
                "score": scores.get(w.url, 0)
            })
            
        data = {
            "timestamp": time.time(),
            "active_workers": sum(1 for w in workers if w.status == "ACTIVE"),
            "total_active_jobs": sum(w.active_jobs for w in workers),
            "queue_size": scheduler.queue.qsize(),
            "workers": worker_data,
            "logs": list(log_buffer.formatted_buffer)[-50:] # Tail 50
        }
        
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.5)

@router.get("/events")
async def sse_endpoint(request: Request):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    workers = cluster.get_snapshot()
    # Sort: Active first, then by name
    workers.sort(key=lambda w: (w.status != "ACTIVE", w.specs.get("worker_name", "")))
    active_count = sum(1 for w in workers if w.status == "ACTIVE")
    total_jobs = sum(w.active_jobs for w in workers)
    scores = {w.url: cluster._calculate_score(w) for w in workers}
    
    context = {
        "request": request,
        "workers": workers,
        "active_workers": active_count,
        "total_active_jobs": total_jobs,
        "queue_size": scheduler.queue.qsize(),
        "logs": list(log_buffer.formatted_buffer),
        "scores": scores,
        "now": time.time(),
        "hostname": os.getenv("HOSTNAME", "odin-master")
    }
    return templates.TemplateResponse("dashboard.html", context)

job_history = collections.deque(maxlen=1000)

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    # 1. Check live active/queued jobs
    if job_id in scheduler.jobs:
        job = scheduler.jobs[job_id]
        return JobResponse(
            job_id=job_id,
            status=job.status,
            result=None,
            worker=job.worker_url if job.worker_url else "pending",
            error=None
        )

    # 2. Check history (completed/failed)
    for job in job_history:
        if job.job_id == job_id:
            return job
            
    raise HTTPException(status_code=404, detail="Job not found")

@router.post("/submit", response_model=JobResponse)
async def submit_job(submission: JobSubmission):
    job_id = str(uuid.uuid4())
    logger.info(f"Received job submission {job_id}")

    # Upload Payload to MinIO
    payload = {
        "code": submission.code,
        "entrypoint": submission.entrypoint,
        "args": submission.args
    }
    payload_path = f"jobs/{job_id}/payload.json"
    
    # Run blocking I/O in executor to avoid freezing the loop
    try:
        if storage:
            logger.info(f"Uploading payload to {payload_path}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: storage.upload_bytes(json.dumps(payload).encode("utf-8"), payload_path)
            )
            logger.info(f"Upload complete: {payload_path}")
    except Exception as e:
        logger.error(f"Failed to upload payload for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Storage upload failed")
    
    future = asyncio.get_event_loop().create_future()
    request = JobRequest(
        id=job_id,
        code="", # Clear code from request to save bandwidth if payload_path is used
        entrypoint=submission.entrypoint,
        args=[], # Clear args from request
        requires_gpu=submission.requires_gpu,
        job_type=submission.job_type,
        payload_path=payload_path
    )
    
    internal_job = InternalJob(request=request, future=future)
    await scheduler.submit(internal_job)
    
    try:
        result_data = await future
        response = JobResponse(
            job_id=job_id,
            status="COMPLETED" if not result_data.get("error") else "FAILED",
            result=result_data.get("result"),
            worker=result_data.get("worker", "unknown"),
            error=result_data.get("error"),
            routing_info=result_data.get("routing_info")
        )
    except Exception as e:
        response = JobResponse(
            job_id=job_id,
            status="FAILED",
            worker="unknown",
            error=str(e)
        )
    
    job_history.append(response)
    return response

@router.get("/logs")
async def get_buffer_logs():
    return list(log_buffer.formatted_buffer)

@router.get("/logs/{node_id}")
async def get_worker_logs(node_id: str, limit: int = 100):
    API_TOKEN = os.getenv("API_TOKEN", "default-insecure-token")
    target_worker = next((w for w in cluster.workers.values() if w.metrics.get("node_id") == node_id), None)
            
    if not target_worker:
        raise HTTPException(status_code=404, detail="Worker not found")
        
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{target_worker.url}/logs", params={"limit": limit}, headers={"Authorization": f"Bearer {API_TOKEN}"})
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch logs")
        return resp.json()
        return resp.json()

# --- Workflow Endpoints ---
from triforce.common.models.workflows import WorkflowRequest, WorkflowResponse

@router.post("/workflows", response_model=Dict[str, Any])
async def submit_workflow(request: WorkflowRequest):
    if not workflow_manager:
        raise HTTPException(status_code=503, detail="Workflow Manager not initialized")
        
    try:
        wf_id = await workflow_manager.submit_workflow(request)
        return {"workflow_id": wf_id, "status": "submitted"}
    except Exception as e:
        logger.exception(f"Failed to submit workflow")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow_status(workflow_id: str):
    if not workflow_manager:
        raise HTTPException(status_code=503, detail="Workflow Manager not initialized")
        
    wf_state = workflow_manager.active_workflows.get(workflow_id)
    
    # If not in memory, check DB (if we implemented full historical loading, which we partially did in _recover)
    if not wf_state:
        # Fallback to DB check
        row = workflow_manager.store.get_workflow(workflow_id)
        if row:
             state_dict = json.loads(row["state"])
             # Return state as response
             # Need to map WorkflowState to WorkflowResponse
             step_statuses = {sid: s["status"] for sid, s in state_dict["step_states"].items()}
             return WorkflowResponse(
                 workflow_id=workflow_id,
                 status=row["status"],
                 step_statuses=step_statuses,
                 created_at=row["created_at"]
             )
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    # Map in-memory state
    step_statuses = {sid: s["status"] for sid, s in wf_state.step_states.items()}
    return WorkflowResponse(
        workflow_id=wf_state.workflow_id,
        status=wf_state.status,
        step_statuses=step_statuses,
        created_at=wf_state.created_at
    )
