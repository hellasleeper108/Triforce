import asyncio
import logging
import json
import time
import math
import statistics
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from triforce.odin.stan.ai_provider import AIProvider
from triforce.odin.stan.memory import MemoryEngine, MemoryType

# --- Data Models ---

class TelemetryPoint(BaseModel):
    timestamp: float
    cpu_usage: float
    gpu_usage: float
    vram_usage_gb: float
    temp_c: float
    tasks_running: int

class Forecast(BaseModel):
    horizon_seconds: int
    predicted_load: float
    failure_probability: float
    risk_factors: List[str]
    reasoning: str

class ActionRecommendation(BaseModel):
    action: str
    urgency: str # "low", "medium", "critical"
    reason: str

# --- 1. Statistical Engine (The "Lizard Brain") ---

class StatisticalForecaster:
    """
    Uses EMA and Linear Regression for fast, cheap trend detection.
    """
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha # For EMA

    def predict_linear(self, history: List[float], steps_ahead: int) -> float:
        """Simple Linear Regression Forecast."""
        if len(history) < 2:
            return history[-1] if history else 0.0
            
        n = len(history)
        xs = list(range(n))
        ys = history
        
        # Calculate slope (m) and intercept (b)
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        
        if denominator == 0:
            return mean_y
            
        m = numerator / denominator
        b = mean_y - (m * mean_x)
        
        target_x = n + steps_ahead
        return (m * target_x) + b

    def detect_spikes_ema(self, history: List[float], threshold_std: float = 2.0) -> bool:
        """Detect if latest value spikes above EMA + N*StdDev."""
        if not history: return False
        
        # Calculate EMA
        ema = history[0]
        for val in history[1:]:
            ema = (val * self.alpha) + (ema * (1 - self.alpha))
            
        if len(history) > 5:
            stdev = statistics.stdev(history)
            if history[-1] > (ema + threshold_std * stdev):
                return True
        return False

# --- 2. Semantic Engine (The "Cortex") ---

class SemanticForecaster:
    """
    Uses LLM + RAG to predict complex, non-linear failure modes.
    """
    
    SYSTEM_PROMPT = """
    You are the PRECOGNITION brain of STAN.
    Analyze telemetry trends and historical context.
    Predict specific failures: OOM, Thermal Throttling, Deadlocks.
    
    Context provided:
    - Recent metrics trend
    - Historical incidents from RAG
    
    Output JSON:
    {
        "failure_prob": 0.0-1.0,
        "risks": ["risk1", "risk2"],
        "recommendation": "Migrate job..."
    }
    """

    def __init__(self, ai: AIProvider, memory: MemoryEngine):
        self.ai = ai
        self.memory = memory

    async def analyze_risk(self, node_name: str, metrics: List[TelemetryPoint]) -> Dict[str, Any]:
        # 1. Summarize Trend
        temps = [m.temp_c for m in metrics]
        vram = [m.vram_usage_gb for m in metrics]
        trend_desc = (
            f"Avg Temp: {statistics.mean(temps):.1f}C (Max: {max(temps)}C). "
            f"VRAM: {statistics.mean(vram):.1f}GB -> {vram[-1]:.1f}GB."
        )
        
        # 2. RAG Lookup
        context = await self.memory.get_context_for_reasoning(
            f"Failure patterns for {node_name} with high temp or VRAM"
        )
        
        # 3. LLM Prediction
        prompt = (
            f"Node: {node_name}\n"
            f"Trend: {trend_desc}\n"
            f"History: {context}\n\n"
            "Predict likelihood of failure in next 10 minutes."
        )
        
        try:
            resp = await self.ai.generate(prompt=prompt, system=self.SYSTEM_PROMPT, json_format=True)
            return json.loads(resp)
        except Exception:
            return {"failure_prob": 0.0, "risks": [], "recommendation": "None"}

# --- 3. The Predictive Brain (Facade) ---

class PredictiveBrain:
    def __init__(self, ai: AIProvider, memory: MemoryEngine):
        self.stats = StatisticalForecaster()
        self.semantic = SemanticForecaster(ai, memory)
        self.history: Dict[str, List[TelemetryPoint]] = {}
        self.logger = logging.getLogger("stan.forecasting")

    def ingest_telemetry(self, node: str, point: TelemetryPoint):
        if node not in self.history:
            self.history[node] = []
        self.history[node].append(point)
        # Keep last 60 points
        if len(self.history[node]) > 60:
            self.history[node].pop(0)

    async def predict_node_state(self, node: str, horizon_seconds: int = 600) -> Forecast:
        """
        Combines statistical projection with semantic risk analysis.
        """
        points = self.history.get(node, [])
        if not points:
            return Forecast(horizon_seconds=horizon_seconds, predicted_load=0, failure_probability=0, risk_factors=[], reasoning="No data")

        # A. Statistical Projection (GPU Usage)
        gpu_hist = [p.gpu_usage for p in points]
        # Assumes 5s intervals. Horizon/5 steps ahead.
        steps = horizon_seconds // 5
        pred_gpu = self.stats.predict_linear(gpu_hist, steps)
        pred_gpu = min(100.0, max(0.0, pred_gpu)) # Clamp

        # B. Semantic Risk Analysis
        risk_data = await self.semantic.analyze_risk(node, points[-12:]) # Last minute context
        
        reasoning = (
            f"Statistical projection indicates GPU load reaching {pred_gpu:.1f}%. "
            f"AI Risk Assessment: {risk_data.get('recommendation')}"
        )

        return Forecast(
            horizon_seconds=horizon_seconds,
            predicted_load=pred_gpu,
            failure_probability=risk_data.get("failure_prob", 0.0),
            risk_factors=risk_data.get("risks", []),
            reasoning=reasoning
        )

    async def recommend_actions(self) -> List[ActionRecommendation]:
        actions = []
        for node in self.history:
            forecast = await self.predict_node_state(node, horizon_seconds=300) # 5 min lookahead
            
            # Heuristic Triggers
            if forecast.failure_probability > 0.7:
                 actions.append(ActionRecommendation(
                     action=f"Drain node {node}", 
                     urgency="critical", 
                     reason=f"High failure prob ({forecast.failure_probability}): {forecast.reasoning}"
                 ))
            elif forecast.predicted_load > 95:
                 actions.append(ActionRecommendation(
                     action=f"Stop scheduling on {node}", 
                     urgency="medium", 
                     reason="projected_saturation"
                 ))
                 
        return actions

# --- Demo Driver ---

async def run_forecasting_demo():
    from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig
    
    # Mock
    class ForecastMock(AIProvider):
        async def generate(self, prompt, system, **kwargs):
            return json.dumps({
                "failure_prob": 0.85, 
                "risks": ["VRAM Exhaustion", "Thermal Throttling"], 
                "recommendation": "Migrate Job ID-99 immediately."
            })
        async def embed(self, t): return []
        async def classify(self, t, l): return l[0]

    ai = ForecastMock(ProviderConfig(type="mock"))
    mem = MemoryEngine(ai, db_path=":memory:")
    brain = PredictiveBrain(ai, mem)
    
    print("--- 1. Generating Synthetic Telemetry ---")
    # Simulate a rising temp trend on 'Loki'
    base_temp = 60.0
    for i in range(20):
        # Temp rises, GPU usage rises
        pt = TelemetryPoint(
             timestamp=time.time() + (i*5),
             cpu_usage=30.0,
             gpu_usage=50.0 + (i * 2.5), # Linear rise to 100
             vram_usage_gb=12.0 + (i * 0.5),
             temp_c=base_temp + (i * 1.5), # Rises to 90
             tasks_running=1
        )
        brain.ingest_telemetry("Loki", pt)
        
    print("--- 2. Predicting Node State (Loki) ---")
    forecast = await brain.predict_node_state("Loki", horizon_seconds=300)
    print(f"Predicted GPU Load (5min): {forecast.predicted_load:.2f}%")
    print(f"Failure Probability: {forecast.failure_probability}")
    print(f"Reasoning: {forecast.reasoning}")
    
    print("\n--- 3. Action Recommendations ---")
    actions = await brain.recommend_actions()
    for a in actions:
        print(f"[{a.urgency.upper()}] {a.action} -> {a.reason}")

if __name__ == "__main__":
    asyncio.run(run_forecasting_demo())
