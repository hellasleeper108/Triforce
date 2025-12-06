
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

    def submit_job(self, func: Callable, args: list, job_type: str = "compute", gpu_required: bool = False) -> Dict[str, Any]:
        """
        Submit a function to the cluster.
        job_type: 'compute' (default), 'gpu_train', 'gpu_infer', 'io_heavy'
        """
        if isinstance(func, str):
            code = func
            entrypoint = "main"
        else:
            code = self._serialize_function(func)
            entrypoint = func.__name__

        # Auto-detect GPU requirement from job_type if not explicitly set
        if job_type in ["gpu_train", "gpu_infer"]:
            gpu_required = True

        payload = {
            "code": code,
            "entrypoint": entrypoint,
            "args": list(args),
            "requires_gpu": gpu_required,
            "job_type": job_type
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
            if resp.status_code == 404:
                return {"status": "NOT_FOUND", "error": "Job ID not found"}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"status": "UNKNOWN", "error": str(e)}

    def get_worker_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all workers (filtered view logic if needed, 
        but API returns list of workers so returning that list matches intent).
        """
        return self.get_cluster_state()

    def get_cluster_state(self) -> List[Dict[str, Any]]:
        """Retrieve the snapshot of all registered workers and their metadata."""
        try:
            resp = self.session.get(f"{self.odin_url}/nodes", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch cluster state: {e}")
            return []

    def fetch_logs(self, worker: str = None) -> List[Dict[str, Any]]:
        """
        Fetch logs.
        If worker (node_id) is provided, fetch specific logs from that worker (via ODIN).
        If worker is None, fetch the central log buffer from ODIN dashboard context logic (needs new endpoint or parsing dashboard? 
        Actually, we updated route to parse logs in `routes.py`, but dashboard is HTML.
        ODIN should expose a JSON endpoint for central logs if we want this.
        However, the user requirement says "Recent logs pulled through ODIN's API".
        I will assume this means using the `/logs/{node_id}` endpoint if worker is set.
        If worker is None, I'll assume they want ALL logs. I'll need to add a `/logs` endpoint to ODIN that returns the central buffer, 
        OR iterate all nodes.
        Let's add a `/logs` (no ID) endpoint to ODIN to return the buffer.
        """
        try:
            url = f"{self.odin_url}/logs" 
            if worker:
                url = f"{self.odin_url}/logs/{worker}"
            
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 404 and not worker:
                # Fallback: if /logs global doesn't exist yet, warn or empty
                return [{"error": "Global logs endpoint not implemented"}]
            
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return [{"error": str(e)}]

    def recommend_best_worker_for(self, task_type: str = "general") -> Optional[str]:
        """
        Recommend best worker Node ID or URL based on criteria.
        task_type: 'gpu' or 'general'
        """
        nodes = self.get_cluster_state()
        active_nodes = [n for n in nodes if n.get("status") == "ACTIVE"]
        
        if not active_nodes:
            return None

        candidates = []
        if task_type.lower() == "gpu":
             candidates = [n for n in active_nodes if n.get("specs", {}).get("gpu_available")]
        else:
             candidates = active_nodes

        if not candidates:
            return None

        # Replicate Scheduler Weight Logic or use ODIN's scores if exposed?
        # ODIN dashboard route has scores, but /nodes API returns workers list usually.
        # Let's inspect /nodes response format. It contains "active_jobs" and metrics.
        # We can calculate weight locally.
        
        def calculate_score(w):
            metrics = w.get("metrics", {})
            cpu = metrics.get("cpu", 0)
            ram = metrics.get("ram", 0)
            gpu = metrics.get("gpu", 0)
            active = w.get("active_jobs", 0)
            return (cpu * 0.4) + (ram * 0.2) + (gpu * 0.2) + (active * 0.2)

        candidates.sort(key=calculate_score)
        return candidates[0]["url"]

