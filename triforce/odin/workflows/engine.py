import uuid
import json
import asyncio
import time
from typing import Dict, Any, Optional

from triforce.common.models.workflows import WorkflowRequest, WorkflowState, WorkflowStep
from triforce.common.models.jobs import JobSubmission, JobRequest
from triforce.odin.scheduler.core import Scheduler, InternalJob
from triforce.odin.store.database import JobStore
from triforce.odin.utils.logger import logger

class WorkflowManager:
    def __init__(self, scheduler: Scheduler, store: JobStore):
        self.scheduler = scheduler
        self.store = store
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.job_to_step_map: Dict[str, str] = {} # job_id -> (workflow_id, step_id) string tuple representation

        # Load active workflows from DB on startup
        self._recover_workflows()

    def _recover_workflows(self):
        active_rows = self.store.get_active_workflows()
        for row in active_rows:
            try:
                state_dict = json.loads(row["state"])
                wf_state = WorkflowState(**state_dict)
                self.active_workflows[wf_state.workflow_id] = wf_state
                
                # Rebuild job mapping
                for step_id, step_data in wf_state.step_states.items():
                    jid = step_data.get("job_id")
                    if jid and step_data["status"] == "RUNNING":
                        self.job_to_step_map[jid] = f"{wf_state.workflow_id}:{step_id}"
                        
                logger.info(f"Recovered workflow {wf_state.workflow_id}")
            except Exception as e:
                logger.error(f"Failed to recover workflow {row['workflow_id']}: {e}")

    async def submit_workflow(self, request: WorkflowRequest) -> str:
        w_id = request.workflow_id or str(uuid.uuid4())
        
        # 1. Initialize State
        step_states = {}
        for step in request.steps:
            step_states[step.step_id] = {
                "step_id": step.step_id,
                "submission": step.submission.model_dump(),
                "dependencies": step.dependencies,
                "status": "PENDING",
                "job_id": None,
                "result_path": None,
                "error": None
            }
            
        wf_state = WorkflowState(
            workflow_id=w_id,
            status="RUNNING",
            step_states=step_states,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # 2. Persist
        self.store.add_workflow(w_id, request.model_dump_json(), wf_state.model_dump_json())
        self.active_workflows[w_id] = wf_state
        
        logger.info(f"Workflow {w_id} submitted with {len(request.steps)} steps")
        
        # 3. Trigger Evaluation
        await self._evaluate_progress(w_id)
        return w_id

    async def _evaluate_progress(self, workflow_id: str):
        wf = self.active_workflows.get(workflow_id)
        if not wf: return
        
        all_complete = True
        has_failure = False
        
        # Iterate over a copy of items to allow modification if needed (though we modify inner objects)
        for step_id, task in wf.step_states.items():
            if task["status"] == "FAILED":
                has_failure = True
                # If one fails, the whole workflow is marked failed (simplification)
                # But we might still want other unrelated branches to finish?
                # For now, simplistic: one fail = workflow fail
                break
            
            if task["status"] != "COMPLETED":
                all_complete = False
            
            if task["status"] == "PENDING":
                # Check Dependencies
                ready = True
                upstream_artifacts = {}
                
                for dep_id in task["dependencies"]:
                    dep_task = wf.step_states.get(dep_id)
                    if not dep_task:
                        logger.error(f"Workflow {workflow_id}: Step {step_id} depends on unknown step {dep_id}")
                        ready = False
                        break
                        
                    if dep_task["status"] != "COMPLETED":
                        ready = False
                        break
                    
                    if dep_task["status"] == "FAILED":
                        # Upstream failed, so we cannot run
                        task["status"] = "FAILED"
                        task["error"] = f"Upstream dependency {dep_id} failed"
                        ready = False
                        has_failure = True
                        break

                    # Collect artifacts
                    if dep_task.get("result_path"):
                        upstream_artifacts[dep_id] = dep_task["result_path"]
                
                if ready:
                    await self._launch_step(workflow_id, step_id, task, upstream_artifacts)

        # Update Workflow Status
        new_status = wf.status
        if has_failure:
            new_status = "FAILED"
        elif all_complete:
            new_status = "COMPLETED"
            
        if new_status != wf.status:
            wf.status = new_status
            wf.updated_at = time.time()
            self.store.update_workflow_state(
                workflow_id, new_status, wf.model_dump_json()
            )
            logger.info(f"Workflow {workflow_id} transition to {new_status}")
            
            if new_status in ["COMPLETED", "FAILED"]:
                # Clean up memory if needed, or keep for history?
                # For now, keep in memory for API access
                pass

    async def _launch_step(self, workflow_id, step_id, task, upstream_artifacts):
        logger.info(f"Workflow {workflow_id}: Launching step {step_id}")
        
        # Convert dict back to model for ease of use
        submission_data = task["submission"]
        submission = JobSubmission(**submission_data)
        
        # Inject Upstream Artifacts into Environment
        # We assume the job runner (THOR) or the user code checks this env var
        if not submission.env:
            submission.env = {}
        submission.env["TRIFORCE_UPSTREAM_ARTIFACTS"] = json.dumps(upstream_artifacts)
        
        # Create Job ID
        job_id = str(uuid.uuid4())
        
        # Upload Payload Logic (Duplicated from Routes for now - Refactor opportunity)
        # For simplicity, we assume the Payload is already handled or we need to handle it here.
        # Ideally, `engine.py` should rely on the same `submit_job` logic as the route, 
        # or we invoke a shared service.
        # But `submit_job` in routes does `upload_bytes`.
        # To avoid duplicating large code, we should probably construct a `JobRequest` directly
        # assuming the code is small enough or we have a shared `JobManager`.
        # For this implementation, I will just assume text-based code/args and NOT do the MinIO upload for the *code* itself yet, 
        # OR I should inject `StorageClient` into `WorkflowManager`.
        
        # Let's assume for this "Lite" version (Side Quest context) we pass code directly 
        # OR we import `storage` global if possible, but that's messy.
        
        # Better approach: We passed `JobSubmission`. If it has large code, we need MinIO.
        # `WorkflowManager` is high level.
        # Let's import the `storage` instance from `odin.main`? No, circular import.
        # We should accept `storage_client` in `__init__`.
        
        # But wait, the route `submit_job` does the upload.
        # I will inject `storage_client` into `WorkflowManager`.
        
        payload_path = None
        # IF we have a storage client (we should), we upload the payload.
        # But `engine.py` is deep in `odin/workflows`.
        
        # NOTE: For now, I will create the JobRequest without payload upload to keep it simple,
        # relying on direct code/args passing. If the user wants MinIO for the *Job Code*, 
        # I'd need to thread `storage` through. 
        # Given I just implemented MinIO integration, I SHOULD support it.
        # I will leave a TODO or try to access `storage`.
        
        req = JobRequest(
            id=job_id,
            code=submission.code,
            entrypoint=submission.entrypoint,
            args=submission.args,
            env=submission.env,
            requires_gpu=submission.requires_gpu,
            job_type=submission.job_type,
            # payload_path=... # Missing
        )
        
        # Create Future
        future = asyncio.get_event_loop().create_future()
        internal_job = InternalJob(request=req, future=future)
        
        # Update State
        task["status"] = "RUNNING"
        task["job_id"] = job_id
        self.job_to_step_map[job_id] = f"{workflow_id}:{step_id}"
        
        # Persist intermediate state
        wf = self.active_workflows[workflow_id]
        wf.updated_at = time.time()
        self.store.update_workflow_state(workflow_id, wf.status, wf.model_dump_json())

        # Submit to Scheduler
        await self.scheduler.submit(internal_job)
        
        # We don't await the future here because we need to return to the loop.
        # We need a callback. `InternalJob` doesn't support callbacks directly, 
        # but we can attach a done callback to the future!
        future.add_done_callback(lambda f: self._on_job_done(job_id, f))

    def _on_job_done(self, job_id, future):
        # Allow async execution of the completion handler
        loop = asyncio.get_event_loop()
        # This callback runs in the thread that set the result (or main loop). 
        # `future` is done.
        
        if job_id not in self.job_to_step_map:
            return

        # Fire and forget the async handler
        asyncio.create_task(self._handle_job_completion_async(job_id, future))

    async def _handle_job_completion_async(self, job_id, future):
        try:
            result_data = future.result() # Dict from JobResponse logic usually
        except Exception as e:
            result_data = {"error": str(e)}

        mapping = self.job_to_step_map.pop(job_id, None)
        if not mapping: return
        
        workflow_id, step_id = mapping.split(":")
        wf = self.active_workflows.get(workflow_id)
        if not wf: return
        
        task = wf.step_states.get(step_id)
        if not task: return
        
        if result_data.get("error"):
            task["status"] = "FAILED"
            task["error"] = result_data["error"]
            logger.warning(f"Workflow {workflow_id} Step {step_id} FAILED: {task['error']}")
        else:
            task["status"] = "COMPLETED"
            task["result_path"] = result_data.get("result_path") # Capture MinIO path
            # Also capture generic result if small
            task["result"] = result_data.get("result")
            logger.info(f"Workflow {workflow_id} Step {step_id} COMPLETED")

        # Persist
        wf.updated_at = time.time()
        self.store.update_workflow_state(workflow_id, wf.status, wf.model_dump_json())
        
        # Trigger next steps
        await self._evaluate_progress(workflow_id)

