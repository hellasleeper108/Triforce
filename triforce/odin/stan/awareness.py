import logging
import time
import json
import statistics
import asyncio
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Imports
from triforce.odin.stan.ai_provider import AIProvider
from triforce.odin.stan.memory import MemoryEngine, MemoryType

# --- Data Models ---

class NodeHealth(BaseModel):
    node_id: str
    status: str
    health_score: float # 0.0 - 1.0
    risk_factors: List[str]
    last_updated: float = Field(default_factory=time.time)

class PredictionReport(BaseModel):
    node_id: str
    failure_probability: float # 0.0 - 1.0
    est_time_to_failure: str # "N/A" or "2h"
    reasoning: str

class AnomalyExplanation(BaseModel):
    summary: str
    root_cause_hypothesis: str
    remediation_suggestion: str
    confidence: float

class ClusterState(BaseModel):
    total_nodes: int
    active_nodes: int
    avg_load: float
    cluster_health: float
    warnings: List[str]
    ai_summary: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

# --- Awareness Brain ---

class AwarenessBrain:
    """
    The 'Eye' of STAN - Enhanced with Cortex.
    Aggregates telemetry, scores health, and uses AI to explain why things are breaking.
    """

    SYSTEM_PROMPT = """
    You are the AWARENESS brain of STAN.
    Analyze cluster telemetry and memory context.
    Identify anomalies, predict failures, and suggest resource redistribution.
    Be concise and technical.
    """

    def __init__(self, registry, ai_provider: AIProvider, memory: MemoryEngine):
        self.registry = registry
        self.ai = ai_provider
        self.memory = memory
        self.logger = logging.getLogger("stan.awareness")
        
        # Telemetry Store (Rolling Window)
        self.metric_history: Dict[str, List[dict]] = {} 
        self.max_history_len = 50

    def ingest_metrics(self, node_id: str, metrics: dict):
        if node_id not in self.metric_history:
            self.metric_history[node_id] = []
        
        entry = {**metrics, "timestamp": time.time()}
        self.metric_history[node_id].append(entry)
        
        if len(self.metric_history[node_id]) > self.max_history_len:
            self.metric_history[node_id].pop(0)

    # --- Heuristic + AI Hybrid Diagnostics ---

    async def detect_anomalies(self) -> Dict[str, AnomalyExplanation]:
        """
        1. Runs fast heuristics.
        2. If anomaly found, calls AI to explain 'Why' using RAG.
        """
        anomalies = {}
        
        # 1. Heuristic Scan
        for node in self.registry.list_all():
            if node.status != "ACTIVE": continue
            
            issues = []
            if getattr(node.metrics, "cpu_usage", 0) > 90: issues.append(f"CPU {node.metrics.cpu_usage}%")
            if getattr(node.metrics, "gpu_temp", 0) > 85: issues.append(f"GPU Temp {node.metrics.gpu_temp}C")
            
            # Simple latency check
            latency = time.time() - node.last_seen
            if latency > 60: issues.append(f"High Latency {int(latency)}s")
            
            if issues:
                # 2. AI Reasoning
                anomalies[node.node_id] = await self._explain_anomaly(node, issues)
        
        return anomalies

    async def _explain_anomaly(self, node, issues: List[str]) -> AnomalyExplanation:
        start_ts = time.time()
        
        # Fetch Context
        context_str = await self.memory.get_context_for_reasoning(f"Anomaly on {node.worker_name}: {issues}")
        
        prompt = (
            f"Node: {node.worker_name} ({node.node_id})\n"
            f"Issues: {issues}\n"
            f"Specs: {node.specs.__dict__}\n"
            f"History Context: {context_str}\n\n"
            "Diagnose the root cause and suggest remediation."
        )
        
        try:
            self.logger.info(f"Diagnosing {node.worker_name}...")
            # Generate JSON explanation
            resp = await self.ai.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT + " Output JSON: {summary, root_cause, suggestion, confidence}",
                json_format=True
            )
            data = json.loads(resp)
            latency = (time.time() - start_ts) * 1000
            self.logger.info(f"Diagnosis complete in {latency:.2f}ms")
            
            return AnomalyExplanation(
                summary=data.get("summary", "Anomaly Detected"),
                root_cause_hypothesis=data.get("root_cause", "Unknown"),
                remediation_suggestion=data.get("suggestion", "Check logs"),
                confidence=data.get("confidence", 0.5)
            )
        except Exception as e:
            self.logger.error(f"AI diag failed: {e}")
            return AnomalyExplanation(
                summary=f"Issues: {issues}",
                root_cause_hypothesis="Heuristic detection only",
                remediation_suggestion="Manual inspection required",
                confidence=1.0
            )

    async def get_predictions(self) -> List[PredictionReport]:
        """
        Predicts node failures based on trends.
        """
        predictions = []
        for node_id, history in self.metric_history.items():
            if len(history) < 10: continue
            
            # Simple Trend: Is temp rising?
            temps = [h.get("gpu_temp", 0) for h in history]
            if len(temps) > 1 and temps[-1] > 80 and temps[-1] > temps[0]:
                predictions.append(PredictionReport(
                    node_id=node_id,
                    failure_probability=0.8,
                    est_time_to_failure="<1h",
                    reasoning="Thermal runaway detected. GPU temp rising consistently."
                ))
        
        return predictions

    async def get_cluster_state(self) -> ClusterState:
        workers = self.registry.list_all()
        active = [w for w in workers if w.status != "OFFLINE"]
        
        # Base Metrics
        total = len(workers)
        alive = len(active)
        avg_load = statistics.mean([w.metrics.cpu_usage for w in active]) if active else 0
        
        health_scores = [self.get_node_health(w.node_id).health_score for w in active]
        cluster_health = statistics.mean(health_scores) if health_scores else 0.0
        
        warnings = []
        if alive < total: warnings.append(f"{total - alive} nodes OFFLINE")
        
        # AI Summary (Optional - expensive, so maybe randomized or cached)
        ai_summary = None
        # In full impl, we might run this periodically, not per request
        
        return ClusterState(
            total_nodes=total,
            active_nodes=alive,
            avg_load=avg_load,
            cluster_health=cluster_health,
            warnings=warnings,
            ai_summary="Cluster nominal." # Placeholder for sync call
        )

    def get_node_health(self, node_id: str) -> NodeHealth:
        # Re-implementing base heuristic logic for speed + consistency
        w = self.registry.get_worker(node_id)
        if not w: return NodeHealth(node_id=node_id, status="UNKNOWN", health_score=0, risk_factors=[])
        
        score = 1.0
        risks = []
        
        if w.status == "OFFLINE":
             return NodeHealth(node_id=node_id, status="OFFLINE", health_score=0, risk_factors=["Offline"])
             
        if getattr(w.metrics, "cpu_usage", 0) > 90:
            score -= 0.2
            risks.append("CPU Saturation")
            
        return NodeHealth(
            node_id=node_id,
            status=w.status,
            health_score=max(0, score),
            risk_factors=risks
        )

# --- Example Driver ---

async def run_awareness_demo():
    from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig
    
    # Mocks
    class MockRegistry:
        def list_all(self):
            return [type("W",(),{
                "node_id":"n1", "worker_name":"Loki", "status":"ACTIVE", 
                "metrics": type("M",(),{"cpu_usage":95, "gpu_temp":88})(),
                "specs": type("S",(),{"model":"MacBook"})(),
                "last_seen": time.time()
            })()]
        def get_worker(self, nid): return self.list_all()[0]

    class MockMemory:
         async def get_context_for_reasoning(self, q): return "History: Loki overheats when running 70B models."
            
    # Setup
    ai = AIProviderFactory.create(ProviderConfig(type="mock"))
    mem = MockMemory()
    brain = AwarenessBrain(MockRegistry(), ai, mem)
    
    print("--- 1. Anomaly AI Diagnosis ---")
    anomalies = await brain.detect_anomalies()
    print(json.dumps({k: v.dict() for k,v in anomalies.items()}, indent=2))
    
    print("\n--- 2. Predictions ---")
    # Inject fake history
    brain.metric_history["n1"] = [{"gpu_temp": 70}, {"gpu_temp": 80}, {"gpu_temp": 88}]
    preds = await brain.get_predictions()
    for p in preds:
        print(f"Prediction: {p.reasoning} (Prob: {p.failure_probability})")

if __name__ == "__main__":
    asyncio.run(run_awareness_demo())
