import re
import uuid
import json
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field

# --- Data Models ---

class ResourceRequirements(BaseModel):
    gpu: bool = False
    min_vram_gb: int = 0
    min_ram_gb: int = 0
    platform: Optional[str] = None # e.g., "darwin", "linux"

class ExecutionStep(BaseModel):
    step_id: str
    action: str
    args: Dict[str, Any] = Field(default_factory=dict)

class TaskNode(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    high_level_action: str
    target_nodes: List[str] = Field(default_factory=list) # e.g., ["thor", "loki", "all"]
    required_resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    execution_steps: List[ExecutionStep] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)

class TaskGraph(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_command: str
    tasks: List[TaskNode]

# --- Parser Logic ---

class CommandParser:
    """
    Parses natural language commands into structured TaskGraphs.
    Uses a hybrid approach:
    1. Rule-based Pattern Matching (for speed and determinism on common tasks)
    2. LLM Fallback (for complex, ambiguous, or creative tasks)
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        # Compile common regex patterns
        self.rules = [
            (r"run (?:a )?(\d+[BM])? ?model on ([\w-]+)", self._parse_run_model),
            (r"send (\w+) a ([\w\s]+) batch", self._parse_send_batch),
            (r"(?:distribute|spread) (?:the )?workload evenly", self._parse_distribute_load),
            (r"run on (?:the )?macbook", self._parse_macbook_target),
            (r"train (?:on )?gpu", self._parse_gpu_training),
        ]

    def parse(self, command: str) -> TaskGraph:
        """Main entry point for parsing."""
        command_lower = command.lower().strip()
        
        # 1. Try Rule-based
        for pattern, handler in self.rules:
            match = re.search(pattern, command_lower)
            if match:
                return handler(match, command)

        # 2. Fallback to LLM
        return self._llm_parse(command_lower)

    # --- Rule Handlers ---

    def _parse_run_model(self, match: re.Match, original: str) -> TaskGraph:
        size = match.group(1) or "unknown"
        target = match.group(2)
        
        resources = ResourceRequirements(gpu=True, min_vram_gb=24 if "70b" in size else 12)
        
        task = TaskNode(
            high_level_action="run_llm_inference",
            target_nodes=[target],
            required_resources=resources,
            execution_steps=[
                ExecutionStep(step_id="pull", action="pull_model", args={"model_size": size}),
                ExecutionStep(step_id="load", action="load_model", args={"quantization": "4bit"}),
                ExecutionStep(step_id="serve", action="start_server", args={"port": 8000})
            ]
        )
        return TaskGraph(original_command=original, tasks=[task])

    def _parse_send_batch(self, match: re.Match, original: str) -> TaskGraph:
        target = match.group(1)
        batch_type = match.group(2)
        
        task = TaskNode(
            high_level_action="process_batch",
            target_nodes=[target],
            required_resources=ResourceRequirements(gpu=False),
            execution_steps=[
                ExecutionStep(step_id="download", action="download_data", args={"source": "s3://bucket/data"}),
                ExecutionStep(step_id="process", action="run_preprocessing", args={"type": batch_type})
            ]
        )
        return TaskGraph(original_command=original, tasks=[task])

    def _parse_distribute_load(self, match: re.Match, original: str) -> TaskGraph:
        task = TaskNode(
            high_level_action="balance_cluster",
            target_nodes=["all"],
            required_resources=ResourceRequirements(),
            execution_steps=[
                ExecutionStep(step_id="analyze", action="get_cluster_metrics", args={}),
                ExecutionStep(step_id="rebalance", action="migrate_jobs", args={"strategy": "even_spread"})
            ]
        )
        return TaskGraph(original_command=original, tasks=[task])

    def _parse_macbook_target(self, match: re.Match, original: str) -> TaskGraph:
        task = TaskNode(
            high_level_action="local_compute",
            target_nodes=["loki"],
            required_resources=ResourceRequirements(platform="darwin"), # Explicitly request Mac
            execution_steps=[
                ExecutionStep(step_id="exec", action="run_script", args={"script": "entrypoint.py"})
            ]
        )
        return TaskGraph(original_command=original, tasks=[task])

    def _parse_gpu_training(self, match: re.Match, original: str) -> TaskGraph:
        task = TaskNode(
            high_level_action="train_model",
            target_nodes=["thor"], # Default to Thor for training
            required_resources=ResourceRequirements(gpu=True, min_vram_gb=40),
            execution_steps=[
                 ExecutionStep(step_id="setup", action="prepare_environment", args={"cuda": "12.0"}),
                 ExecutionStep(step_id="train", action="run_training_loop", args={"epochs": 10})
            ]
        )
        return TaskGraph(original_command=original, tasks=[task])

    # --- LLM Fallback (Simulated) ---

    def _llm_parse(self, command: str) -> TaskGraph:
        """
        In a real system, this would call OpenAI/Gemini APIs.
        Here we mock it with a simple heuristic for demonstration.
        """
        # Heuristic Mock for "complex" commands
        if "sleep" in command:
            task = TaskNode(
                high_level_action="system_sleep",
                target_nodes=["all"],
                execution_steps=[ExecutionStep(step_id="sleep", action="sleep", args={"duration": 10})]
            )
        elif "deploy" in command:
             task = TaskNode(
                high_level_action="deploy_service",
                target_nodes=["odin"],
                execution_steps=[ExecutionStep(step_id="deploy", action="k8s_apply", args={"manifest": "deployment.yaml"})]
            )
        else:
             # Generic Fallback
             task = TaskNode(
                high_level_action="unknown_command",
                target_nodes=["odin"],
                execution_steps=[ExecutionStep(step_id="log", action="log_error", args={"msg": "Could not parse intent"})]
            )
            
        return TaskGraph(original_command=command, tasks=[task])

# --- Example Generation ---

def generate_examples():
    parser = CommandParser()
    
    commands = [
        "Run a 70B model on Thor",
        "Send Loki a preprocessing batch",
        "Distribute the workload evenly",
        "Run on the macbook",
        "Train on GPU",
        "Run a 13B model on Odin",
        "Spread workload evenly",
        "Send Thor a training batch", # Uses regex generalize
        "Sleep for a bit", # Falls back to LLM heuristic
        "Deploy the new stack" # Falls back to LLM heuristic
    ]
    
    examples = []
    for cmd in commands:
        graph = parser.parse(cmd)
        examples.append(json.loads(graph.model_dump_json()))
    
    return examples

if __name__ == "__main__":
    exs = generate_examples()
    print(json.dumps(exs, indent=2))
