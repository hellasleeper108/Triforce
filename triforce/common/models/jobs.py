from pydantic import BaseModel
from typing import List, Any, Optional, Dict

class JobSubmission(BaseModel):
    code: str
    entrypoint: str = "main"
    args: List[Any] = []
    requires_gpu: bool = False
    job_type: str = "compute" # compute, gpu_train, gpu_infer, io_heavy
    env: Optional[Dict[str, str]] = None

class JobRequest(BaseModel):
    id: str
    code: str
    entrypoint: str
    args: List[Any]
    retries: int = 0
    requires_gpu: bool = False
    job_type: str = "compute"
    payload_path: Optional[str] = None
    result_path: Optional[str] = None
    env: Optional[Dict[str, str]] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Any = None
    worker: str
    error: Optional[str] = None
    routing_info: Optional[str] = None
    result_path: Optional[str] = None
