import time
import os
import asyncio
import httpx
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

from triforce.odin.utils.logger import logger

MAX_CONCURRENCY_PER_WORKER = int(os.getenv("MAX_CONCURRENCY", 4))
API_TOKEN = os.getenv("API_TOKEN", "default-insecure-token")

class RegistrationData(BaseModel):
    node_id: str
    url: str
    worker_name: str # alias for hostname
    ip: str
    port: int
    gpu_available: bool
    cpu_cores: int
    gpu_mem_total: int
    memory_gb: float
    capabilities: List[str] = []
    gpus: List[Dict[str, Any]] = []
    worker_class: Optional[str] = None # Optional override by worker

@dataclass
class Worker:
    url: str
    node_id: str = ""
    status: str = "UNKNOWN"
    last_seen: float = 0.0
    active_jobs: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    specs: Dict[str, Any] = field(default_factory=dict)
    worker_class: str = "cpu"

    @property
    def is_active(self):
        return self.status == "ACTIVE"

    @property
    def has_capacity(self):
        return self.active_jobs < MAX_CONCURRENCY_PER_WORKER

# Import job models for type hinting in get_best_worker if needed
# from triforce.common.models.jobs import JobRequest

class ClusterManager:
    def __init__(self):
        self.workers: Dict[str, Worker] = {}

    def register(self, data: RegistrationData):
        logger.info(f"Worker heartbeat: {data.node_id} ({data.url})")
        specs = {
            "cpu_cores": data.cpu_cores,
            "gpu_mem_total": data.gpu_mem_total,
            "memory_gb": data.memory_gb,
            "gpu_available": data.gpu_available,
            "worker_name": data.worker_name,
            "ip": data.ip,
            "port": data.port
        }
        
        # Infer worker class if not provided
        w_class = data.worker_class
        if not w_class:
            if data.gpu_available:
                w_class = "gpu"
            elif data.cpu_cores < 4:
                w_class = "light"
            else:
                w_class = "cpu"
        
        if data.url not in self.workers:
            self.workers[data.url] = Worker(
                url=data.url, 
                node_id=data.node_id,
                status="ACTIVE", 
                last_seen=time.time(),
                capabilities=data.capabilities,
                gpus=data.gpus,
                specs=specs,
                worker_class=w_class
            )
        else:
            w = self.workers[data.url]
            w.last_seen = time.time()
            w.status = "ACTIVE"
            w.node_id = data.node_id
            w.capabilities = data.capabilities
            w.gpus = data.gpus
            w.specs = specs
            w.worker_class = w_class

    def get_snapshot(self) -> List[Worker]:
        return list(self.workers.values())

    def _calculate_score(self, w: Worker) -> float:
        # 1. System Metrics (New Format)
        cpu = w.metrics.get("cpu", 0.0)
        mem = w.metrics.get("ram", 0.0)
        
        # 2. GPU Load (Direct value or specific field)
        gpu_load = w.metrics.get("gpu", 0.0)
        
        # 3. Job Saturation
        job_load = (w.active_jobs / MAX_CONCURRENCY_PER_WORKER) * 100
        
        # Weighted Score:
        # CPU: 1.0, RAM: 1.0, GPU: 1.5, Jobs: 2.0 (High penalty for job queue depth)
        score = (cpu * 1.0) + (mem * 1.0) + (gpu_load * 1.5) + (job_load * 2.0)
        return score

    def get_best_worker(self, job_request: Any = None) -> Optional[str]:
        # job_request is strictly typed as JobRequest usually, but Any to avoid circ imports for now
        candidates = [
            w for w in self.workers.values() 
            if w.is_active and w.has_capacity
        ]
        
        if job_request and getattr(job_request, 'requires_gpu', False):
            candidates = [w for w in candidates if "gpu" in w.capabilities]
            if not candidates:
                logger.debug("No GPU workers available for GPU job")
                return None
        
        if not candidates:
            return None
            
        candidates.sort(key=self._calculate_score)
        best = candidates[0]
        score = self._calculate_score(best)
        req_type = "GPU" if job_request and getattr(job_request, 'requires_gpu', False) else "Standard"
        logger.debug(f"Selected {best.url} for {req_type} job (Score: {score:.1f})")
        return best.url

    async def update_health(self):
        while True:
            now = time.time()
            for w in self.workers.values():
                if now - w.last_seen > 15:
                    if w.status != "OFFLINE":
                        logger.warning(f"Worker {w.url} timed out")
                        w.status = "OFFLINE"
            
            active_workers = [w for w in self.workers.values() if w.status == "ACTIVE"]
            tasks = [self._poll_metrics(w) for w in active_workers]
            if tasks:
                await asyncio.gather(*tasks)
            
            await asyncio.sleep(5)

            # Prune dead workers (Offline > 5 mins)
            prune_threshold = now - 300
            dead_urls = [
                w.url for w in self.workers.values() 
                if w.status == "OFFLINE" and w.last_seen < prune_threshold
            ]
            for url in dead_urls:
                logger.info(f"Pruning dead worker: {url}")
                del self.workers[url]


    async def _poll_metrics(self, worker: Worker):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                headers = {
                    "Authorization": f"Bearer {API_TOKEN}",
                    "X-Role": "controller"
                }
                resp = await client.get(f"{worker.url}/metrics", headers=headers)
                if resp.status_code == 200:
                    worker.metrics = resp.json()
                    worker.last_seen = time.time()
        except Exception:
            pass
