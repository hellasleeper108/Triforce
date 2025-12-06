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

    def _calculate_weight(self, worker) -> float:
        # User defined formula:
        # weight = (cpu_usage * 0.4) + (ram_usage * 0.2) + (gpu_usage * 0.2) + (active_jobs * 0.2)
        
        cpu = worker.metrics.get("cpu", 0.0)
        ram = worker.metrics.get("ram", 0.0)
        gpu = worker.metrics.get("gpu", 0.0)
        
        # Use real-time active_jobs from worker state, not stale metrics
        active = worker.active_jobs
        
        weight = (cpu * 0.4) + (ram * 0.2) + (gpu * 0.2) + (active * 0.2)
        return weight

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
                
                # Filter for GPU if required
                if job.request.requires_gpu:
                     candidates = [w for w in candidates if "gpu" in w.capabilities]
                
                if candidates:
                    candidates.sort(key=self._calculate_weight)
                    best_worker = candidates[0]
                    
                    # Log selected worker and weight
                    weight = self._calculate_weight(best_worker)
                    logger.debug(f"Selected {best_worker.url} (Weight: {weight:.2f})")
                    
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
