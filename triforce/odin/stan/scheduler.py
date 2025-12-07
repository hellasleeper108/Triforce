import json
import time
import uuid
import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# Mocking imports if they don't exist in environment yet, 
# in real usage these would be:
# from triforce.odin.stan.parser import TaskGraph, TaskNode, ResourceRequirements
# from triforce.odin.stan.registry import WorkerRegistry, WorkerManifest

# --- Data Models (Re-declared for standalone usage in this snippet) ---

class ResourceRequirements(BaseModel):
    gpu: bool = False
    min_vram_gb: int = 0
    min_ram_gb: int = 0
    platform: Optional[str] = None 

class TaskNode(BaseModel):
    task_id: str
    high_level_action: str
    target_nodes: List[str]
    required_resources: ResourceRequirements
    dependencies: List[str]

class TaskGraph(BaseModel):
    request_id: str
    original_command: str
    tasks: List[TaskNode]

class WorkerSpecs(BaseModel):
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_mem_total: int = 0
    platform: str = "linux"

class WorkerMetrics(BaseModel):
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    gpu_usage: float = 0.0
    active_jobs: int = 0

class WorkerManifest(BaseModel):
    node_id: str
    worker_name: str
    status: str
    specs: WorkerSpecs
    metrics: WorkerMetrics

class Assignment(BaseModel):
    task_id: str
    node_id: str
    worker_name: str
    score: float
    strategy: str # "direct", "greedy", "fallback"

class DispatchPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    assignments: List[Assignment]
    unassigned: List[str] = []
    created_at: float = Field(default_factory=time.time)

# --- Scheduler Implementation ---

class Scheduler:
    """
    Core Scheduling Engine for STAN.
    """

    def __init__(self, registry):
        self.registry = registry
        self.logger = logging.getLogger("stan.scheduler")

    def schedule(self, graph: TaskGraph) -> DispatchPlan:
        assignments = []
        unassigned = []
        
        # Topological sort could go here, but for now we iterate
        for task in graph.tasks:
            assigned = self._assign_task(task)
            if assigned:
                assignments.append(assigned)
            else:
                unassigned.append(task.task_id)
                
        return DispatchPlan(
            request_id=graph.request_id,
            assignments=assignments,
            unassigned=unassigned
        )

    def _assign_task(self, task: TaskNode) -> Optional[Assignment]:
        workers = self.registry.list_all()
        candidates = [w for w in workers if w.status == "ACTIVE"]

        # 1. Direct Target Filtering
        if task.target_nodes and "all" not in task.target_nodes:
            candidates = [w for w in candidates if w.worker_name.lower() in task.target_nodes or w.node_id in task.target_nodes]
            if not candidates:
                # If specific target requested but offline/missing -> Fail assignment
                self.logger.warning(f"Target {task.target_nodes} not found/active for task {task.task_id}")
                return None
            
            # If explicit target, pick best among them (usually just one)
            best = self._pick_greedy(task, candidates)
            return Assignment(task_id=task.task_id, node_id=best.node_id, worker_name=best.worker_name, score=1.0, strategy="direct")

        # 2. Resource Filtering
        candidates = self._filter_by_resources(task, candidates)
        if not candidates:
            # Fallback Check (Soft Constraints?)
            # For now, just fail
            return None

        # 3. Model Sharding Policy Check
        # If task needs massive VRAM (e.g. > 80GB) and no single node has it,
        # we might need pipelining. 
        # Here we just simple check:
        if task.required_resources.min_vram_gb > 80:
             # Assume we need multi-node or refuse
             self.logger.info("Task requires >80GB VRAM. Triggering Sharding Logic (Not Implemented).")
             return None

        # 4. Greedy Allocation (Score & Pick)
        best_node = self._pick_greedy(task, candidates)
        
        # Calculate Score
        score = self._score_node(task, best_node)
        
        return Assignment(
            task_id=task.task_id,
            node_id=best_node.node_id,
            worker_name=best_node.worker_name,
            score=score,
            strategy="greedy"
        )

    def _filter_by_resources(self, task: TaskNode, candidates: List[WorkerManifest]) -> List[WorkerManifest]:
        req = task.required_resources
        filtered = []
        for w in candidates:
            # Platform Check
            if req.platform and w.specs.platform != req.platform:
                continue
            
            # GPU Check
            if req.gpu and not w.specs.gpu_available:
                continue
                
            # VRAM Check (MB)
            if req.min_vram_gb > 0:
                if w.specs.gpu_mem_total < (req.min_vram_gb * 1024):
                    continue
            
            # RAM Check
            if req.min_ram_gb > 0:
                if w.specs.memory_gb < req.min_ram_gb:
                    continue
                    
            filtered.append(w)
        return filtered

    def _pick_greedy(self, task: TaskNode, candidates: List[WorkerManifest]) -> WorkerManifest:
        # Sort by Score descending
        return sorted(candidates, key=lambda w: self._score_node(task, w), reverse=True)[0]

    def _score_node(self, task: TaskNode, w: WorkerManifest) -> float:
        """
        Cost Scoring:
        Score = (ResourceFit * 0.4) + (Availability * 0.4) + (PerformancePrior * 0.2)
        """
        # 1. Resource Fit (How much headroom?)
        # We prefer nodes where the task fits 'comfortably' but doesn't waste massive capacity?
        # Actually for scheduling, usually 'Least Loaded' is key.
        
        # Availability (Inverse of Load)
        cpu_load = w.metrics.cpu_usage / 100.0
        gpu_load = w.metrics.gpu_usage / 100.0 if w.specs.gpu_available else 0.0
        mem_load = w.metrics.ram_usage / 100.0
        
        avg_load = (cpu_load + mem_load + (gpu_load if task.required_resources.gpu else 0)) / (3 if task.required_resources.gpu else 2)
        availability_score = 1.0 - avg_load
        
        # Penalty for 'Busy' status even if metrics are low (e.g. reserved)
        if w.status == "BUSY": 
            availability_score *= 0.5

        # Performance Multiplier (Static Tiering)
        perf_score = 1.0
        if w.worker_name.lower() == "odin": perf_score = 1.2 # Master is fastest usually
        if w.worker_name.lower() == "thor": perf_score = 1.1 # Worker is strong
        if w.worker_name.lower() == "loki": perf_score = 0.8 # Edge is slower

        # Platform Affinity
        # If task doesn't specify platform but candidate matches system preference
        affinity = 1.0
        
        final_score = (availability_score * 0.7) + (perf_score * 0.3)
        return round(final_score, 3)

# --- Mock Registry for Examples ---

class MockRegistry:
    def list_all(self):
        return [
            WorkerManifest(
                node_id="n1", worker_name="odin", status="ACTIVE",
                specs=WorkerSpecs(cpu_cores=32, memory_gb=96, gpu_available=True, gpu_mem_total=16384, platform="linux"),
                metrics=WorkerMetrics(cpu_usage=10, gpu_usage=20, active_jobs=2)
            ),
            WorkerManifest(
                node_id="n2", worker_name="thor", status="ACTIVE",
                specs=WorkerSpecs(cpu_cores=16, memory_gb=32, gpu_available=True, gpu_mem_total=12288, platform="linux"),
                metrics=WorkerMetrics(cpu_usage=80, gpu_usage=90, active_jobs=5) # Busy
            ),
            WorkerManifest(
                node_id="n3", worker_name="loki", status="ACTIVE",
                specs=WorkerSpecs(cpu_cores=8, memory_gb=16, gpu_available=False, gpu_mem_total=0, platform="darwin"),
                metrics=WorkerMetrics(cpu_usage=5, active_jobs=0) # Idle
            )
        ]

# --- Examples ---

def run_examples():
    registry = MockRegistry()
    scheduler = Scheduler(registry)
    
    print("=== Scheduling Examples ===\n")

    # Example 1: Heavy GPU Training
    # Expect: Odin (Thor is busy, Loki has no GPU)
    t1 = TaskNode(
        task_id="t1", high_level_action="train", target_nodes=[], dependencies=[],
        required_resources=ResourceRequirements(gpu=True, min_vram_gb=10)
    )
    plan1 = scheduler.schedule(TaskGraph(request_id="r1", original_command="Train model", tasks=[t1]))
    print(f"Ex 1 (GPU Train): Assigned to {plan1.assignments[0].worker_name} (Score: {plan1.assignments[0].score})")

    # Example 2: Mac-specific Task
    # Expect: Loki
    t2 = TaskNode(
        task_id="t2", high_level_action="local_build", target_nodes=[], dependencies=[],
        required_resources=ResourceRequirements(platform="darwin")
    )
    plan2 = scheduler.schedule(TaskGraph(request_id="r2", original_command="Run on Mac", tasks=[t2]))
    print(f"Ex 2 (Mac Task):  Assigned to {plan2.assignments[0].worker_name} (Score: {plan2.assignments[0].score})")

    # Example 3: General Compute (Load Balanced)
    # Expect: Loki (Lowest Load) or Odin (Perf). 
    # Loki Load=5%, Odin Load=10%. Loki avail=0.95, Odin avail=0.9.
    # Loki Perf=0.8, Odin Perf=1.2.
    # Loki Score = 0.95*0.7 + 0.8*0.3 = 0.665 + 0.24 = 0.905
    # Odin Score = 0.9*0.7 + 1.2*0.3 = 0.63 + 0.36 = 0.99
    # Winner: Odin (Performance outweighs slight load diff)
    t3 = TaskNode(
        task_id="t3", high_level_action="process", target_nodes=[], dependencies=[],
        required_resources=ResourceRequirements()
    )
    plan3 = scheduler.schedule(TaskGraph(request_id="r3", original_command="Compute stuff", tasks=[t3]))
    print(f"Ex 3 (Gen Comp):  Assigned to {plan3.assignments[0].worker_name} (Score: {plan3.assignments[0].score})")

if __name__ == "__main__":
    run_examples()
