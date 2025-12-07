import asyncio
import logging
import json
import time
import copy
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Simulation Primitives ---

class SimulatedTask(BaseModel):
    id: str
    cpu_cores: float
    ram_gb: float
    gpu_load: float
    vram_gb: float
    duration_seconds: int
    priority: int = 1

class SimulatedNode(BaseModel):
    id: str
    total_cpu: float
    total_ram: float
    total_gpu: float
    total_vram: float
    
    used_cpu: float = 0
    used_ram: float = 0
    used_gpu: float = 0
    used_vram: float = 0
    
    running_tasks: List[SimulatedTask] = Field(default_factory=list)
    status: str = "healthy" # "healthy", "overloaded", "crashed"

    def allocate(self, task: SimulatedTask) -> bool:
        if (self.used_cpu + task.cpu_cores <= self.total_cpu and
            self.used_ram + task.ram_gb <= self.total_ram and
            self.used_gpu + task.gpu_load <= self.total_gpu and
            self.used_vram + task.vram_gb <= self.total_vram):
            
            self.used_cpu += task.cpu_cores
            self.used_ram += task.ram_gb
            self.used_gpu += task.gpu_load
            self.used_vram += task.vram_gb
            self.running_tasks.append(task)
            return True
        return False

    def release(self, task: SimulatedTask):
        if task in self.running_tasks:
            self.used_cpu -= task.cpu_cores
            self.used_ram -= task.ram_gb
            self.used_gpu -= task.gpu_load
            self.used_vram -= task.vram_gb
            self.running_tasks.remove(task)

class SimulationState(BaseModel):
    time_offset: int = 0
    nodes: Dict[str, SimulatedNode] = {}
    pending_tasks: List[SimulatedTask] = []
    completed_tasks: List[SimulatedTask] = []
    failed_tasks: List[SimulatedTask] = []
    events_log: List[str] = []

# --- World Model Engine ---

class WorldModel:
    """
    The internal simulator for STAN.
    """
    def __init__(self):
        self.logger = logging.getLogger("stan.world_model")

    def create_snapshot(self, real_nodes: List[Dict[str, Any]]) -> SimulationState:
        """
        Hydrates simulation from real world telemetry.
        """
        sim_nodes = {}
        for n in real_nodes:
            sim_nodes[n["id"]] = SimulatedNode(
                id=n["id"],
                total_cpu=n.get("cpu_cores", 8),
                total_ram=n.get("ram_gb", 32),
                total_gpu=n.get("gpu_percent", 100), # Abstract capacity
                total_vram=n.get("vram_gb", 16)
            )
        return SimulationState(nodes=sim_nodes)

    def step(self, state: SimulationState, seconds: int = 1):
        """
        Advances the simulation by N seconds.
        """
        state.time_offset += seconds
        
        # 1. Process Running Tasks (Decrement Duration)
        for nid, node in state.nodes.items():
            if node.status == "crashed":
                continue
                
            # Random Failure Check (Probabilistic Engine)
            if self._check_failure(node):
                node.status = "crashed"
                state.events_log.append(f"T+{state.time_offset}: Node {nid} CRASHED (Overload/Random)")
                # Fail all tasks
                for t in node.running_tasks:
                    state.failed_tasks.append(t)
                node.running_tasks = [] 
                node.used_cpu = 0 # Reset usage but node is dead
                continue

            finished = []
            for t in node.running_tasks:
                t.duration_seconds -= seconds
                if t.duration_seconds <= 0:
                    finished.append(t)
            
            for t in finished:
                node.release(t)
                state.completed_tasks.append(t)
                state.events_log.append(f"T+{state.time_offset}: Task {t.id} completed on {nid}")

        # 2. Schedule Pending Tasks (Naive Greedy)
        # In a real World Model, this would use the Planner brain.
        remaining_pending = []
        for t in state.pending_tasks:
            allocated = False
            # Sort nodes by available RAM (Simple heuristic)
            sorted_nodes = sorted(state.nodes.values(), key=lambda n: n.total_ram - n.used_ram, reverse=True)
            
            for node in sorted_nodes:
                if node.status == "healthy" and node.allocate(t):
                    state.events_log.append(f"T+{state.time_offset}: Task {t.id} started on {node.id}")
                    allocated = True
                    break
            
            if not allocated:
                remaining_pending.append(t)
        
        state.pending_tasks = remaining_pending

    def _check_failure(self, node: SimulatedNode) -> bool:
        """
        Probabilistic failure function.
        Risk increases with load.
        """
        load_factor = (node.used_cpu / node.total_cpu) + (node.used_ram / node.total_ram)
        base_risk = 0.0001 # Very low base
        if load_factor > 1.5: # Heavily loaded
            base_risk = 0.01 # 1% chance per second
        elif load_factor > 1.8:
            base_risk = 0.05
            
        return random.random() < base_risk

    def run_scenario(self, initial_state: SimulationState, horizon_seconds: int) -> SimulationState:
        """
        Runs a What-If scenario (Monte Carlo run).
        """
        sim_state = copy.deepcopy(initial_state)
        for _ in range(horizon_seconds):
            self.step(sim_state, 1)
        return sim_state

# --- Demo Driver ---

async def run_world_model_demo():
    print("--- 1. Initializing Cluster Snapshot ---")
    wm = WorldModel()
    
    real_data = [
        {"id": "odin", "cpu_cores": 16, "ram_gb": 64, "vram_gb": 0},
        {"id": "thor", "cpu_cores": 32, "ram_gb": 128, "vram_gb": 24}, # GPU Node
        {"id": "loki", "cpu_cores": 8, "ram_gb": 32, "vram_gb": 0}
    ]
    
    snapshot = wm.create_snapshot(real_data)
    
    print("--- 2. What-If Scenario: Heavy Load ---")
    # Add a massive batch of tasks
    new_tasks = [
        SimulatedTask(id=f"job-{i}", cpu_cores=4, ram_gb=16, gpu_load=0, vram_gb=0, duration_seconds=10)
        for i in range(10)
    ]
    # Add one GPU heavy task
    new_tasks.append(SimulatedTask(id="gpu-job-1", cpu_cores=8, ram_gb=32, gpu_load=80, vram_gb=20, duration_seconds=20))
    
    snapshot.pending_tasks = new_tasks
    
    print(f"Simulating {len(new_tasks)} tasks over 30 seconds...")
    result_state = wm.run_scenario(snapshot, 30)
    
    print("\n--- Simulation Results ---")
    print(f"Completed Tasks: {len(result_state.completed_tasks)}")
    print(f"Failed Tasks:    {len(result_state.failed_tasks)}")
    print(f"Pending Tasks:   {len(result_state.pending_tasks)}")
    
    print("\nEvent Log (Subset):")
    for event in result_state.events_log[:10]:
        print(f" {event}")
    if len(result_state.events_log) > 10: print(" ...")

    # Inspect Thor (GPU Node) final state
    thor_node = result_state.nodes["thor"]
    print(f"\nThor Final State: RAM {thor_node.used_ram}/{thor_node.total_ram} GB, VRAM {thor_node.used_vram}/{thor_node.total_vram} GB")

if __name__ == "__main__":
    asyncio.run(run_world_model_demo())
