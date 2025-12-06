import time
import io
import sys
import traceback
from pydantic import BaseModel
from typing import Any, List

from triforce.thor.utils.logger import logger
from triforce.common.models.jobs import JobRequest

class JobResult(BaseModel):
    job_id: str
    status: str
    result: Any = None
    stdout: str
    stderr: str
    duration_ms: float

def execute_python_code(job: JobRequest) -> JobResult:
    start_time = time.time()
    logger.info(f"Starting execution of job {job.id}")
    
    capture_stdout = io.StringIO()
    capture_stderr = io.StringIO()
    
    local_scope = {}
    global_scope = {
        "__builtins__": __builtins__,
    }

    status = "COMPLETED"
    result = None
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = capture_stdout
    sys.stderr = capture_stderr

    try:
        exec(job.code, global_scope, local_scope)
        
        if job.entrypoint not in local_scope:
            raise ValueError(f"Entrypoint '{job.entrypoint}' not found in executed code.")
        
        entry_func = local_scope[job.entrypoint]
        result = entry_func(*job.args)
        
    except Exception as e:
        status = "FAILED"
        traceback.print_exc()
        logger.error(f"Job {job.id} failed: {e}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    duration = (time.time() - start_time) * 1000
    
    return JobResult(
        job_id=job.id,
        status=status,
        result=result,
        stdout=capture_stdout.getvalue(),
        stderr=capture_stderr.getvalue(),
        duration_ms=duration
    )
