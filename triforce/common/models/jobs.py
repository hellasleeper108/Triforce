from pydantic import BaseModel
from typing import List, Any, Optional

class JobSubmission(BaseModel):
    code: str
    entrypoint: str = "main"
    args: List[Any] = []
    requires_gpu: bool = False

class JobRequest(BaseModel):
    id: str
    code: str
    entrypoint: str
    args: List[Any]
    retries: int = 0
    requires_gpu: bool = False

class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Any = None
    worker: str
    error: Optional[str] = None
