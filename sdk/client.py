
import requests
import inspect
import textwrap
import os
import types
from typing import Any, Dict, List, Optional, Callable, Union

class STANClusterClient:
    """
    Client for interacting with the Stan Cluster (ODIN Master).
    Supports job submission, status querying, log fetching, and topology inspection.
    """
    def __init__(self, odin_url=None, token=None):
        self.odin_url = (odin_url or os.getenv("ODIN_URL", "http://localhost:8080")).rstrip("/")
        self.token = token or os.getenv("API_TOKEN", "default-insecure-token")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _serialize_function(self, func: Callable) -> str:
        source = textwrap.dedent(inspect.getsource(func))
        return source

    def submit(self, func_or_code: Union[Callable, str], *args, gpu: Optional[bool] = None) -> Dict[str, Any]:
        """
        Submit a function or code string to the cluster.
        Blocks until completion by default (ODIN behavior).
        
        Args:
            gpu: If True, requires GPU worker. If None, auto-detects based on code content.
        """
        if isinstance(func_or_code, str):
            code = func_or_code
        else:
            code = self._serialize_function(func_or_code)
            
        # GPU Auto-detection
        requires_gpu = gpu
        if requires_gpu is None:
            keywords = ["cuda", "torch", "tensorflow", "pynvml", "cupy", "nvidia"]
            requires_gpu = any(k in code.lower() for k in keywords)
        
        payload = {
            "code": code,
            "entrypoint": "main" if isinstance(func_or_code, str) else func_or_code.__name__,
            "args": list(args),
            "requires_gpu": requires_gpu
        }
        
        try:
            resp = self.session.post(f"{self.odin_url}/submit", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"status": "FAILED", "error": str(e)}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Query the status and result of a specific job."""
        try:
            resp = self.session.get(f"{self.odin_url}/jobs/{job_id}", timeout=5)
            # Handle 404 gracefully?
            if resp.status_code == 404:
                return {"status": "NOT_FOUND", "error": "Job ID not found"}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"status": "UNKNOWN", "error": str(e)}

    def get_cluster_topology(self) -> List[Dict[str, Any]]:
        """Retrieve the list of all registered workers and their metadata."""
        try:
            resp = self.session.get(f"{self.odin_url}/nodes", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch topology: {e}")
            return []

    def fetch_worker_metrics(self, node_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Fetch metrics for a specific worker or all workers."""
        nodes = self.get_cluster_topology()
        if node_id:
            for n in nodes:
                if n.get("node_id") == node_id:
                    return n.get("metrics", {})
            return {}
        return [n.get("metrics", {}) for n in nodes]

    def fetch_logs(self, worker_node_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent logs from a specific worker."""
        try:
            resp = self.session.get(f"{self.odin_url}/logs/{worker_node_id}", params={"limit": limit}, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch logs: {e}")
            return []

    def get_optimal_gpu_worker(self) -> Optional[Dict[str, Any]]:
        """
        Detect which worker is optimal for ML/GPU tasks.
        Criteria: Has GPU, Lowest GPU Temperature, Lowest Job Count.
        """
        nodes = self.get_cluster_topology()
        gpu_nodes = []
        
        for n in nodes:
            # Check if node has GPU capabilities
            has_gpu = "gpu" in n.get("capabilities", []) or len(n.get("gpus", [])) > 0
            if has_gpu and n.get("status") == "ACTIVE":
                gpu_nodes.append(n)
        
        if not gpu_nodes:
            return None
            
        # Sort by: Active Jobs (Asc), then GPU Temp (Asc)
        def sort_key(n):
            jobs = n.get("active_jobs", 0)
            gpus = n.get("metrics", {}).get("gpus", [])
            temp = gpus[0]["temperature_c"] if gpus else 100
            return (jobs, temp)
            
        gpu_nodes.sort(key=sort_key)
        return gpu_nodes[0]

# Alias for backward compatibility if needed
STANCluster = STANClusterClient
