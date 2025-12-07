import threading
import os
import time
import requests
from pydantic import BaseModel
from typing import List, Dict, Any
import psutil

from triforce.thor.utils.logger import logger
from triforce.thor.metrics.collector import get_gpu_specs

# Globals managed by main, but registration logic is here
class RegistrationData(BaseModel):
    node_id: str
    url: str
    worker_name: str
    ip: str
    port: int
    gpu_available: bool
    cpu_cores: int
    worker_class: str = None
    memory_gb: float
    gpu_mem_total: int = 0
    capabilities: List[str] = []
    gpus: List[Dict[str, Any]] = []

def heartbeat_loop(node_id, port, api_token):
    logger.info("Starting registration loop...")
    odin_url = os.getenv("ODIN_URL", "http://localhost:8080")
    ip = os.getenv("HOST_IP", "127.0.0.1")
    my_url = f"http://{ip}:{port}"
    hostname = os.getenv("HOSTNAME", "thor-node")
    
    import psutil # Ensure psutil is available here
    
    while True:
        try:
            gpus, has_gpu, total_vram = get_gpu_specs()
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
            
            payload = RegistrationData(
                node_id=node_id,
                url=my_url,
                worker_name=hostname,
                ip=ip,
                port=port,
                gpu_available=has_gpu,
                cpu_cores=psutil.cpu_count(logical=True),
                gpu_mem_total=total_vram,
                memory_gb=mem_gb,
                capabilities=["python", "gpu" if has_gpu else "cpu"],
                gpus=gpus
            )
            
            headers = {
                "Authorization": f"Bearer {api_token}",
                "X-Role": "worker"
            }
            
            requests.post(f"{odin_url}/register", json=payload.model_dump(), headers=headers, timeout=2)
            # logger.debug(f"Heartbeat sent for {hostname}")
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            
        time.sleep(5)
