import asyncio
import os
import httpx
from dataclasses import dataclass

from triforce.odin.utils.logger import logger
from triforce.odin.controllers.worker import ClusterManager
from triforce.common.models.jobs import JobRequest

MAX_RETRIES = 3
API_TOKEN = os.getenv("API_TOKEN", "default-insecure-token")

@dataclass
class InternalJob:
    request: JobRequest
    future: asyncio.Future
    retries: int = 0
    status: str = "QUEUED"
    worker_url: str = None
    created_at: float = 0.0

from triforce.odin.store.database import JobStore
import json

class Scheduler:
    def __init__(self, cluster: ClusterManager):
        self.cluster = cluster
        self.queue: asyncio.Queue[InternalJob] = asyncio.Queue()
        self.jobs: dict[str, InternalJob] = {} # Live jobs registry
        self.store = JobStore()
        self._recover_jobs()

    def _recover_jobs(self):
        recoverable = self.store.get_recoverable_jobs()
        if recoverable:
            logger.info(f"Found {len(recoverable)} jobs to recover from DB")
            for job_data in recoverable:
                # Reset to queued in DB
                self.store.reset_job_to_queued(job_data['job_id'])
                
                # Re-create internal job structure to put back in queue
                # We need to deserialize request_payload
                try:
                    payload = json.loads(job_data['request_payload'])
                    req = JobRequest(**payload)
                    future = asyncio.get_event_loop().create_future()
                    
                    internal_job = InternalJob(
                        request=req,
                        future=future,
                        retries=0,
                        status="QUEUED"
                    )
                    self.jobs[req.id] = internal_job
                    self.queue.put_nowait(internal_job)
                    logger.info(f"Recovered job {req.id} to QUEUE")
                except Exception as e:
                    logger.error(f"Failed to recover job {job_data.get('job_id')}: {e}")

    async def submit(self, job: InternalJob):
        import time
        job.created_at = time.time()
        job.status = "QUEUED"
        
        # Persist Initial State
        self.store.add_job(
            job.request.id, 
            job.request.job_type, 
            job.request.model_dump()
        )
        
        self.jobs[job.request.id] = job
        await self.queue.put(job)
        logger.info(f"Job {job.request.id} queued. (Queue Size: {self.queue.qsize()})")

    def _calculate_weight(self, worker, job: InternalJob = None) -> tuple[float, str]:
        # 1. Base Load Score (Lower is better)
        cpu = worker.metrics.get("cpu", 0.0)
        ram = worker.metrics.get("ram", 0.0)
        gpu = worker.metrics.get("gpu", 0.0)
        disk_io = worker.metrics.get("disk_io_mbps", 0.0)
        net_io = worker.metrics.get("net_io_mbps", 0.0)
        active = worker.active_jobs
        
        load_score = (cpu * 1.0) + (ram * 1.0) + (gpu * 1.0) + (active * 5.0)
        reasons = []

        # Adjust for IO Heavy jobs
        if job and job.request.job_type == "io_heavy":
            # For IO heavy, look at Disk/Net IO directly
            # Penalize heavily if IO is already high (> 50 MB/s?)
            io_penalty = (disk_io * 5.0) + (net_io * 5.0)
            load_score += io_penalty
            reasons.append(f"IO_Load(Disk:{disk_io:.1f},Net:{net_io:.1f})")
        else:
            reasons.append(f"Load(CPU:{cpu},RAM:{ram})")

        # 2. Worker Class Penalties (Soft constraints)
        class_penalty = 0.0
        w_class = getattr(worker, "worker_class", "cpu")
        
        if job:
            j_type = job.request.job_type
            
            if j_type == "gpu_train":
                # stricter than soft-penalty, this shouldn't happen if filtering is correct
                # but adding for safety
                if w_class != "gpu":
                     class_penalty += 10000.0
                     reasons.append("TypeMismatch:NeedGPU")
                else:
                    reasons.append("TypeMatch:GPU")
            
            elif j_type == "gpu_infer":
                 # Soft preference for GPU
                 if w_class != "gpu":
                     class_penalty += 500.0 # Prefer GPU, but allow CPU if GPU is swamped (score > 500)
                     reasons.append("Penalty:PreferGPU")
                 else:
                    reasons.append("TypeMatch:GPU")
            
            elif j_type == "compute":
                # Generic compute: prefer CPU workers to save GPUs
                if w_class == "gpu":
                    class_penalty += 50.0 
                    reasons.append("Penalty:SaveGPU")
        
        final_score = load_score + class_penalty
        return final_score, ", ".join(reasons)

    async def run(self):
        logger.info("Scheduler started.")
        while True:
            job = await self.queue.get()
            
            while True:
                # Find best worker
                candidates = [
                    w for w in self.cluster.workers.values() 
                    if w.is_active and w.has_capacity
                ]
                
                # 1. Hard Constraints Filtering
                if job.request.requires_gpu or job.request.job_type == "gpu_train":
                     # "gpu_train" -> GPU workers only
                     candidates = [w for w in candidates if "gpu" in w.capabilities]
                
                if candidates:
                    # Calculate weights and reasons for all candidates
                    scored_candidates = []
                    for w in candidates:
                        weight, reason = self._calculate_weight(w, job)
                        scored_candidates.append((w, weight, reason))
                    
                    # Sort by weighted score (Lower is better)
                    scored_candidates.sort(key=lambda x: x[1])
                    
                    best_worker, weight, decision_reason = scored_candidates[0]
                    
                    # Log selected worker and weight
                    logger.debug(f"Selected {best_worker.url} for {job.request.job_type} (Weight: {weight:.2f}, Reason: {decision_reason})")
                    
                    best_worker.active_jobs += 1
                    
                    job.status = "RUNNING"
                    job.worker_url = best_worker.url
                    # Update DB
                    self.store.update_job(job.request.id, "RUNNING", worker_url=best_worker.url)

                    # Store decision reason in the job object (requires InternalJob update or piggyback)
                    # We can pass it to _execute
                    
                    logger.info(f"Dispatching {job.request.id} -> {best_worker.url}")
                    asyncio.create_task(self._execute(best_worker, job, decision_reason))
                    break
                else:
                    await asyncio.sleep(1)

    async def _execute(self, worker, job: InternalJob, decision_reason: str):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "Authorization": f"Bearer {API_TOKEN}",
                    "X-Role": "controller"
                }
                resp = await client.post(
                    f"{worker.url}/work", 
                    json=job.request.model_dump(),
                    headers=headers
                )
                resp.raise_for_status()
                result = resp.json()
                
                job.status = "COMPLETED"
                self.store.update_job(job.request.id, "COMPLETED", result=result)
                
                # Remove from live registry when done (or keep for short history?)
                # We currently have a separate history deque in API. 
                # Let's keep it in registry until explicitly cleared or just rely on API to cleanup/move to history.
                # Currently API moves to history. We'll delete here?
                # Actually, API awaits future. So we keep it here until future is done.
                
                if not job.future.done():
                    job.future.set_result({
                        "result": result, 
                        "worker": worker.url,
                        "error": None,
                        "routing_info": decision_reason
                    })
        except Exception as e:
            logger.error(f"Job {job.request.id} failed on {worker.url}: {e}")
            worker.status = "OFFLINE"
            
            if job.retries < MAX_RETRIES:
                job.retries += 1
                job.status = "QUEUED" # Back to queue
                job.worker_url = None
                self.store.update_job(job.request.id, "QUEUED", worker_url=None) # Reset in DB
                
                logger.warning(f"Retrying job {job.request.id}")
                await self.queue.put(job)
            else:
                job.status = "FAILED"
                self.store.update_job(job.request.id, "FAILED", error=str(e))
                
                if not job.future.done():
                    job.future.set_result({
                        "result": None,
                        "worker": worker.url,
                        "error": f"Max retries exceeded. Last error: {str(e)}"
                    })
        finally:
            worker.active_jobs -= 1
            self.queue.task_done()
            
            # Cleanup from registry if finished
            if job.status in ["COMPLETED", "FAILED"]:
                 self.jobs.pop(job.request.id, None)
