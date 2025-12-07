import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Mocking Registry/Manifest access for standalone compiling
# from triforce.odin.stan.registry import WorkerRegistry, WorkerManifest, WorkerMetrics

# --- Data Models ---

class NodeHealth(BaseModel):
    node_id: str
    status: str
    health_score: float # 0.0 to 1.0 (1.0 = Healthy, 0.0 = Dead)
    risk_factors: List[str] = []
    last_updated: float = Field(default_factory=time.time)

class ClusterState(BaseModel):
    total_nodes: int
    active_nodes: int
    total_cores: int
    total_memory_gb: float
    total_vram_gb: float
    avg_cpu_load: float
    avg_gpu_load: float
    cluster_health: float
    warnings: List[str] = []
    timestamp: float = Field(default_factory=time.time)

class AnomalyReport(BaseModel):
    anomalies_detected: int
    details: List[str]
    recommendations: List[str]

# --- Awareness System ---

class AwarenessSystem:
    """
    The 'Eye' of STAN. Aggregates telemetry, scores health, and detects anomalies.
    """

    def __init__(self, registry):
        self.registry = registry
        self.logger = logging.getLogger("stan.awareness")
        
        # In-memory history for predictive analytics (rolling window)
        self.metric_history: Dict[str, List[dict]] = {} 
        self.max_history_len = 50

    def ingest_metrics(self, node_id: str, metrics: dict):
        """Called whenever a heartbeat arrives."""
        if node_id not in self.metric_history:
            self.metric_history[node_id] = []
        
        entry = {**metrics, "timestamp": time.time()}
        self.metric_history[node_id].append(entry)
        
        # Prune
        if len(self.metric_history[node_id]) > self.max_history_len:
            self.metric_history[node_id].pop(0)

    # --- State Queries ---

    def get_cluster_state(self) -> ClusterState:
        workers = self.registry.list_all()
        active = [w for w in workers if w.status != "OFFLINE"]
        
        if not workers:
            return ClusterState(
                total_nodes=0, active_nodes=0, total_cores=0, total_memory_gb=0,
                total_vram_gb=0, avg_cpu_load=0, avg_gpu_load=0, cluster_health=0, 
                warnings=["Cluster is empty"]
            )

        total_nodes = len(workers)
        active_nodes = len(active)
        total_cores = sum(w.specs.cpu_cores for w in active)
        total_mem = sum(w.specs.memory_gb for w in active)
        total_vram = sum(w.specs.gpu_mem_total for w in active) / 1024.0 # MB -> GB
        
        # Calculate Averages (avoid Div/0)
        avg_cpu = sum(w.metrics.cpu_usage for w in active) / active_nodes if active_nodes else 0
        avg_gpu = sum(w.metrics.gpu_usage for w in active) / active_nodes if active_nodes else 0
        
        # Aggregate Risk
        node_healths = [self.get_node_health(w.node_id).health_score for w in active]
        cluster_health = statistics.mean(node_healths) if node_healths else 0.0

        warnings = []
        if active_nodes < total_nodes:
            warnings.append(f"{total_nodes - active_nodes} nodes are OFFLINE.")
        if cluster_health < 0.7:
            warnings.append("Cluster health is DEGRADED.")

        return ClusterState(
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            total_cores=total_cores,
            total_memory_gb=total_mem,
            total_vram_gb=total_vram,
            avg_cpu_load=avg_cpu,
            avg_gpu_load=avg_gpu,
            cluster_health=cluster_health,
            warnings=warnings
        )

    def get_node_health(self, node_id: str) -> NodeHealth:
        worker = self.registry.get_worker(node_id)
        if not worker:
            return NodeHealth(node_id=node_id, status="UNKNOWN", health_score=0.0)

        score = 1.0
        risks = []

        # 1. Liveness
        if worker.status == "OFFLINE":
            score = 0.0
            risks.append("Node offline")
            return NodeHealth(node_id=node_id, status="OFFLINE", health_score=score, risk_factors=risks)

        # 2. Staleness (missed heartbeats)
        latency = time.time() - worker.last_seen
        if latency > 30: # >2 Missed beats
            score -= 0.3
            risks.append(f"High Latency ({int(latency)}s)")
        
        # 3. Resource Saturation
        if worker.metrics.cpu_usage > 90:
            score -= 0.2
            risks.append("CPU Saturated (>90%)")
        
        if worker.metrics.ram_usage > 95:
            score -= 0.3
            risks.append("RAM Critical (>95%)")
            
        if worker.specs.gpu_available:
            if worker.metrics.gpu_temp > 85:
                score -= 0.2
                risks.append(f"GPU Overheating ({worker.metrics.gpu_temp}C)")

        return NodeHealth(
            node_id=node_id,
            status=worker.status,
            health_score=max(0.0, score),
            risk_factors=risks
        )

    # --- Analytics ---

    def detect_anomalies(self) -> AnomalyReport:
        details = []
        recommendations = []
        
        # 1. Check for zombie nodes (Active but silent)
        for w in self.registry.list_all():
             latency = time.time() - w.last_seen
             if w.status != "OFFLINE" and latency > 60:
                 details.append(f"Node {w.worker_name} marked ACTIVE but silent for {int(latency)}s (Zombie?)")
                 recommendations.append(f"Restart Thor service on {w.worker_name}")
        
        # 2. Check for resource imbalances (Simple deviation)
        actives = [w for w in self.registry.list_all() if w.status == "ACTIVE"]
        if len(actives) > 1:
            loads = [w.metrics.cpu_usage for w in actives]
            mean_load = statistics.mean(loads)
            stdev_load = statistics.stdev(loads) if len(loads) > 1 else 0
            
            for w in actives:
                if w.metrics.cpu_usage > (mean_load + 2 * stdev_load) and w.metrics.cpu_usage > 50:
                    details.append(f"Node {w.worker_name} CPU load ({w.metrics.cpu_usage}%) significantly higher than cluster mean ({mean_load:.1f}%)")
                    recommendations.append("Consider 'Distribute Workload' command.")

        return AnomalyReport(
            anomalies_detected=len(details),
            details=details,
            recommendations=recommendations
        )

    def recommend_rebalance(self) -> Optional[str]:
        # Simple heuristic wrapper
        report = self.detect_anomalies()
        if "Consider 'Distribute Workload' command." in report.recommendations:
             return "Hotspots detected. Recommendation: Trigger auto-rebalance."
        return None

# --- Mock Implementation for Demo ---

class MockRegistry:
    class MockWorker:
        def __init__(self, nid, name, status, cpu, ram, gpu, gpu_temp, last_seen_delta):
            self.node_id = nid
            self.worker_name = name
            self.status = status
            self.specs = type("Specs", (), {"cpu_cores": 16, "memory_gb": 32, "gpu_available": gpu, "gpu_mem_total": 12000})()
            self.metrics = type("Metrics", (), {"cpu_usage": cpu, "ram_usage": ram, "gpu_usage": 50, "gpu_temp": gpu_temp})()
            self.last_seen = time.time() - last_seen_delta

    def list_all(self):
        return [
            self.MockWorker("n1", "odin", "ACTIVE", 20, 30, True, 65, 5),
            self.MockWorker("n2", "thor", "ACTIVE", 95, 40, True, 88, 5), # Hot & High CPU
            self.MockWorker("n3", "loki", "ACTIVE", 10, 20, False, 0, 70), # Zombie latency
        ]
    
    def get_worker(self, nid):
        for w in self.list_all():
            if w.node_id == nid: return w
        return None

# --- Sample Run ---

def generate_report():
    sys = AwarenessSystem(MockRegistry())
    
    # 1. Cluster State
    state = sys.get_cluster_state()
    print("=== Cluster State ===")
    print(json.dumps(state.model_dump(), indent=2))
    
    # 2. Anomalies
    print("\n=== Anomalies ===")
    anoms = sys.detect_anomalies()
    print(json.dumps(anoms.model_dump(), indent=2))
    
    # 3. Node Health
    print("\n=== Detailed Health ===")
    for nid in ["n1", "n2", "n3"]:
        h = sys.get_node_health(nid)
        status_icon = "🟢" if h.health_score > 0.8 else ("🟡" if h.health_score > 0.5 else "🔴")
        print(f"{status_icon} Node {nid}: Score {h.health_score:.2f} | Risks: {h.risk_factors}")

if __name__ == "__main__":
    generate_report()
