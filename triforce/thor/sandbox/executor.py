import time
import io
import sys
import traceback
from pydantic import BaseModel
from typing import Any, List

from triforce.thor.utils.logger import logger
from triforce.common.models.jobs import JobRequest

import json
from triforce.common.models.jobs import JobResult as BaseJobResult 
# Note: Changing return type to use common JobResult or update local definition?
# The local JobResult is Pydantic.
# Let's align with what main expects. main expects JobResult.
# We also need to add result_path to JobResult definition here if it's local.

class JobResult(BaseModel):
    job_id: str
    status: str
    result: Any = None
    stdout: str
    stderr: str
    duration_ms: float
    result_path: Any = None # Optional[str]

def execute_python_code(job: JobRequest, storage=None) -> JobResult:
    start_time = time.time()
    logger.info(f"Starting execution of job {job.id}")
    
    code = job.code
    entrypoint = job.entrypoint
    args = job.args
    
    # 1. Download Payload if specified
    if job.payload_path and storage:
        try:
            logger.info(f"Downloading payload from {job.payload_path}")
            payload_bytes = storage.download_bytes(job.payload_path)
            payload_data = json.loads(payload_bytes.decode("utf-8"))
            
            code = payload_data.get("code")
            entrypoint = payload_data.get("entrypoint")
            args = payload_data.get("args")
        except Exception as e:
            logger.error(f"Failed to download payload: {e}")
            return JobResult(
                job_id=job.id,
                status="FAILED",
                result=None,
                stdout="",
                stderr=f"Payload download failed: {str(e)}",
                duration_ms=0.0
            )

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
        exec(code, global_scope, local_scope)
        
        if entrypoint not in local_scope:
            raise ValueError(f"Entrypoint '{entrypoint}' not found in executed code.")
        
        entry_func = local_scope[entrypoint]
        result = entry_func(*args)
        
    except Exception as e:
        status = "FAILED"
        traceback.print_exc()
        logger.error(f"Job {job.id} failed: {e}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    duration = (time.time() - start_time) * 1000
    
    # 2. Upload Result if successful and storage available
    result_path = None
    if status == "COMPLETED" and storage:
        try:
            result_path = f"jobs/{job.id}/result.json"
            result_payload = {
                "job_id": job.id,
                "result": result,
                "duration_ms": duration
            }
            # We might want to clear 'result' from the response if it's large and stored in S3
            # For now, keep both for backward compat unless user requested optimization.
            # User request: "upload results". "ODIN must store metadata linking job_id to object paths".
            
            storage.upload_bytes(json.dumps(result_payload).encode("utf-8"), result_path)
            logger.info(f"Uploaded result to {result_path}")
            
            # Optimization: Clear result from body if uploaded?
            # Let's keep it for now to avoid breaking existing clients that expect result body.
        except Exception as e:
            logger.error(f"Failed to upload result: {e}")
            # Don't fail the job if upload fails, but maybe note it?
            capture_stderr.write(f"\nResult upload failed: {e}")
    
    return JobResult(
        job_id=job.id,
        status=status,
        result=result,
        stdout=capture_stdout.getvalue(),
        stderr=capture_stderr.getvalue(),
        duration_ms=duration,
        result_path=result_path
    )
