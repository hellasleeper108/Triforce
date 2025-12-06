from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
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

# Globals (Injected from main)
cluster = None
scheduler = None

@router.post("/register")
def register_worker(data: RegistrationData):
    cluster.register(data)
    return {"status": "registered"}

@router.get("/nodes")
async def get_nodes():
    return cluster.get_snapshot()

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    workers = cluster.get_snapshot()
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
    
    future = asyncio.get_event_loop().create_future()
    request = JobRequest(
        id=job_id,
        code=submission.code,
        entrypoint=submission.entrypoint,
        args=submission.args,
        requires_gpu=submission.requires_gpu
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
            error=result_data.get("error")
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
