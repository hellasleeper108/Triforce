import json
import time
import logging
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path

# --- Data Models ---

class WorkerSpecs(BaseModel):
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_mem_total: int = 0  # MB
    platform: str = "linux" # linux, darwin

class WorkerMetrics(BaseModel):
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_temp: float = 0.0
    disk_io: float = 0.0
    net_io: float = 0.0

class WorkerManifest(BaseModel):
    node_id: str
    worker_name: str
    url: str
    specs: WorkerSpecs
    status: str = "ACTIVE" # ACTIVE, BUSY, OFFLINE
    capabilities: List[str] = Field(default_factory=list)
    metrics: WorkerMetrics = Field(default_factory=WorkerMetrics)
    last_seen: float = Field(default_factory=time.time)

    @validator("status")
    def validate_status(cls, v):
        allowed = ["ACTIVE", "BUSY", "OFFLINE"]
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return v

# --- Registry Implementation ---

class WorkerRegistry:
    """
    In-memory registry for STAN workers with optional persistence.
    Enforces hardware profiles for known nodes.
    """
    
    # Hardware Truths (Enforced Profiles)
    KNOWN_PROFILES = {
        "odin": {
            "gpu_model": "RTX 4080 Super",
            "gpu_mem_total": 16384, # 16GB
            "memory_gb": 96.0,
            "gpu_available": True
        },
        "thor": {
            "gpu_model": "RTX 3080 Ti",
            "gpu_mem_total": 12288, # 12GB
            "memory_gb": 32.0,
            "gpu_available": True
        },
        "loki": {
            "gpu_model": "None (Unified)",
            "gpu_mem_total": 0,
            "memory_gb": 16.0,
            "gpu_available": False, # Treat as CPU-only for scheduling preference
            "platform": "darwin"
        }
    }

    def __init__(self, persistence_path: Optional[str] = None):
        self.workers: Dict[str, WorkerManifest] = {}
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.logger = logging.getLogger("stan.registry")
        
        if self.persistence_path and self.persistence_path.exists():
            self._load()

    # --- Core Management ---

    def add_worker(self, manifest_data: dict) -> WorkerManifest:
        """Register a new worker or update existing one."""
        try:
            # 1. Enforce known hardware profiles based on name
            name = manifest_data.get("worker_name", "").lower()
            if "odin" in name:
                self._apply_profile(manifest_data, "odin")
            elif "thor" in name:
                self._apply_profile(manifest_data, "thor")
            elif "loki" in name:
                self._apply_profile(manifest_data, "loki")

            # 2. Validate
            worker = WorkerManifest(**manifest_data)
            worker.last_seen = time.time()
            
            # 3. Store
            self.workers[worker.node_id] = worker
            self.logger.info(f"Registered worker: {worker.worker_name} ({worker.node_id})")
            
            self._save()
            return worker
            
        except Exception as e:
            self.logger.error(f"Failed to register worker: {e}")
            raise e

    def remove_worker(self, node_id: str):
        if node_id in self.workers:
            del self.workers[node_id]
            self._save()

    def update_status(self, node_id: str, status: str, metrics: dict = None):
        worker = self.workers.get(node_id)
        if worker:
            worker.status = status
            worker.last_seen = time.time()
            if metrics:
                worker.metrics = WorkerMetrics(**metrics)
            self._save()

    def get_worker(self, node_id: str) -> Optional[WorkerManifest]:
        return self.workers.get(node_id)
    
    def list_all(self) -> List[WorkerManifest]:
        return list(self.workers.values())

    # --- Query Capabilities ---

    def get_workers_by_role(self, role: str) -> List[WorkerManifest]:
        """Filter by role/capability (e.g., 'gpu', 'python')."""
        return [w for w in self.workers.values() if role in w.capabilities and w.status != "OFFLINE"]

    def get_best_gpu_node(self) -> Optional[WorkerManifest]:
        """Return the ACTIVE node with the most free VRAM and lowest GPU usage."""
        gpu_nodes = [w for w in self.workers.values() 
                     if w.specs.gpu_available and w.status == "ACTIVE"]
        
        if not gpu_nodes:
            return None
            
        # Score = VRAM (high is good) - Usage (high is bad)
        # Simple heuristic: Maximize VRAM, minimize Load
        def score(w):
            vram = w.specs.gpu_mem_total
            usage = w.metrics.gpu_usage
            return vram * (1 - (usage / 100.0))
            
        return max(gpu_nodes, key=score)

    def get_cpu_heavy_nodes(self) -> List[WorkerManifest]:
        """Return nodes suitable for CPU work (e.g. Loki or idle Master)."""
        # Favor nodes ensuring low CPU load
        nodes = [w for w in self.workers.values() if w.status == "ACTIVE"]
        return sorted(nodes, key=lambda w: w.metrics.cpu_usage)

    def get_nodes_for_model_size(self, param_size_b: float) -> List[WorkerManifest]:
        """
        Estimate memory requirements (4-bit quantization approx).
        70B params ~ 35-40GB VRAM.
        13B params ~ 8-10GB VRAM.
        7B params ~ 5GB VRAM.
        """
        required_vram_gb = param_size_b * 0.7  # Crude 4-bit heuristic
        required_vram_mb = required_vram_gb * 1024
        
        candidates = []
        for w in self.workers.values():
            if w.specs.gpu_available:
                if w.specs.gpu_mem_total >= required_vram_mb:
                    candidates.append(w)
            elif w.specs.platform == "darwin" and w.specs.memory_gb >= required_vram_gb:
                # Mac Unified Memory fallback
                candidates.append(w)
                
        return candidates

    # --- Internals ---

    def _apply_profile(self, data: dict, profile_key: str):
        profile = self.KNOWN_PROFILES[profile_key]
        if "specs" not in data:
            data["specs"] = {}
        
        data["specs"]["gpu_available"] = profile["gpu_available"]
        data["specs"]["memory_gb"] = profile["memory_gb"]
        data["specs"]["gpu_mem_total"] = profile["gpu_mem_total"]
        if "platform" in profile:
            data["specs"]["platform"] = profile["platform"]

    def _save(self):
        if not self.persistence_path:
            return
        try:
            with open(self.persistence_path, "w") as f:
                dump = {k: w.dict() for k, w in self.workers.items()}
                json.dump(dump, f, indent=2)
        except Exception as e:
            self.logger.error(f"Persistence error: {e}")

    def _load(self):
        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    self.workers[k] = WorkerManifest(**v)
        except Exception as e:
            self.logger.error(f"Load error: {e}")
