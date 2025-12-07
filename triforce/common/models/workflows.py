from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from triforce.common.models.jobs import JobSubmission

class WorkflowStep(BaseModel):
    step_id: str = Field(..., description="Unique identifier for the step within the workflow")
    submission: JobSubmission = Field(..., description="Job submission details for this step")
    dependencies: List[str] = Field(default_factory=list, description="List of step_ids that this step depends on")

class WorkflowRequest(BaseModel):
    workflow_id: Optional[str] = Field(None, description="Optional custom ID for the workflow")
    steps: List[WorkflowStep] = Field(..., description="List of steps in the workflow")

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    step_statuses: Dict[str, str]
    error: Optional[str] = None
    created_at: float

class WorkflowState(BaseModel):
    workflow_id: str
    status: str # PENDING, RUNNING, COMPLETED, FAILED
    step_states: Dict[str, Dict[str, Any]] # step_id -> {status, job_id, result_path, error}
    created_at: float
    updated_at: float
