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

class Scheduler:
    def __init__(self, cluster: ClusterManager):
        self.cluster = cluster
        self.queue: asyncio.Queue[InternalJob] = asyncio.Queue()
        self.jobs: dict[str, InternalJob] = {} # Live jobs registry

    async def submit(self, job: InternalJob):
        import time
        job.created_at = time.time()
        job.status = "QUEUED"
        
        self.jobs[job.request.id] = job
        await self.queue.put(job)
        logger.info(f"Job {job.request.id} queued. (Queue Size: {self.queue.qsize()})")

    def _calculate_weight(self, worker, job: InternalJob = None) -> float:
        # 1. Base Load Score (Lower is better)
        cpu = worker.metrics.get("cpu", 0.0)
        ram = worker.metrics.get("ram", 0.0)
        gpu = worker.metrics.get("gpu", 0.0)
        active = worker.active_jobs
        
        # Penalize saturation heavily
        load_score = (cpu * 1.0) + (ram * 1.0) + (gpu * 1.0) + (active * 5.0)
        
        # Adjust for IO Heavy jobs
        if job and job.request.job_type == "io_heavy":
            # For IO heavy, concurrency is the enemy. Penalize active jobs massively.
            load_score = (cpu * 0.5) + (ram * 0.5) + (active * 20.0)

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
            
            elif j_type == "gpu_infer":
                 # Soft preference for GPU
                 if w_class != "gpu":
                     class_penalty += 500.0 # Prefer GPU, but allow CPU if GPU is swamped (score > 500)
            
            elif j_type == "compute":
                # Generic compute: prefer CPU workers to save GPUs
                if w_class == "gpu":
                    class_penalty += 50.0 
        
        return load_score + class_penalty

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
                    # Sort by weighted score (Lower is better)
                    candidates.sort(key=lambda w: self._calculate_weight(w, job))
                    best_worker = candidates[0]
                    
                    # Log selected worker and weight
                    weight = self._calculate_weight(best_worker, job)
                    logger.debug(f"Selected {best_worker.url} for {job.request.job_type} (Weight: {weight:.2f})")
                    
                    best_worker.active_jobs += 1
                    
                    job.status = "RUNNING"
                    job.worker_url = best_worker.url
                    
                    logger.info(f"Dispatching {job.request.id} -> {best_worker.url}")
                    asyncio.create_task(self._execute(best_worker, job))
                    break
                else:
                    await asyncio.sleep(1)

    async def _execute(self, worker, job: InternalJob):
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
                # Remove from live registry when done (or keep for short history?)
                # We currently have a separate history deque in API. 
                # Let's keep it in registry until explicitly cleared or just rely on API to cleanup/move to history.
                # Currently API moves to history. We'll delete here?
                # Actually, API awaits future. So we keep it here until future is done.
                
                if not job.future.done():
                    job.future.set_result({
                        "result": result, 
                        "worker": worker.url,
                        "error": None
                    })
        except Exception as e:
            logger.error(f"Job {job.request.id} failed on {worker.url}: {e}")
            worker.status = "OFFLINE"
            
            if job.retries < MAX_RETRIES:
                job.retries += 1
                job.status = "QUEUED" # Back to queue
                job.worker_url = None
                
                logger.warning(f"Retrying job {job.request.id}")
                await self.queue.put(job)
            else:
                job.status = "FAILED"
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
