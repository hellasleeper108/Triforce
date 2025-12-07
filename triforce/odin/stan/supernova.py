import asyncio
import logging
import json
import time
from typing import Dict, Any, List

# --- Subsystem Imports ---
from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig
from triforce.odin.stan.memory import MemoryEngine, MemoryType
from triforce.odin.stan.autonomy import AutonomyManager, AutonomyLevel, ActionRisk
from triforce.odin.stan.world_model import WorldModel, SimulatedTask
from triforce.odin.stan.automation import AutomationEngine
from triforce.odin.stan.forecasting import PredictiveBrain, TelemetryPoint
from triforce.odin.stan.skills import SkillManager
from triforce.odin.stan.metacognition import MetacognitionEngine
from triforce.odin.stan.assembly import AssemblyEngine, PipelineStep
# Agents
from triforce.odin.stan.agents.impl import ShardMasterAgent, ScribeAgent, OptimizerAgent, SentinelAgent

class STANSupernova:
    """
    The Unified Intelligence Layer for STAN.
    Integrates all cognitive, predictive, and autonomous subsystems.
    """

    def __init__(self, use_mock: bool = True):
        # 1. Logging
        logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
        self.logger = logging.getLogger("stan.supernova")
        
        # 2. Base AI
        config = ProviderConfig(type="mock" if use_mock else "openai")
        self.ai = AIProviderFactory.create(config)
        
        # 3. Core Subsystems
        self.memory = MemoryEngine(self.ai, db_path=":memory:" if use_mock else "stan.db")
        self.autonomy = AutonomyManager(initial_level=AutonomyLevel.ENGINEER)
        self.world = WorldModel()
        self.automation = AutomationEngine()
        self.forecasting = PredictiveBrain(self.ai, self.memory)
        self.skills = SkillManager(self.ai, self.memory)
        self.metacognition = MetacognitionEngine(self.ai)
        self.assembly = AssemblyEngine(self.ai)
        
        # 4. Agent Swarm
        self._init_agents()

    def _init_agents(self):
        self.agents = [
            ShardMasterAgent(self, "ShardMaster", "Distribute compute", self.ai, self.memory),
            ScribeAgent(self, "Scribe", "Log history", self.ai, self.memory),
            OptimizerAgent(self, "Optimizer", "Maximize efficiency", self.ai, self.memory),
            SentinelAgent(self, "Sentinel", "Guard cluster", self.ai, self.memory)
        ]

    # --- Primary Interaction API ---

    async def act(self, command: str):
        """
        The OODA Loop: Observe, Orient, Decide, Act.
        """
        self.logger.info(f"\n[COMMAND] User: '{command}'")

        # 1. Metacognition: Safety Check
        self.logger.info("[1] Metacognition: Critiquing intent...")
        critique = await self.metacognition.critique_plan(command)
        if critique.score < 0.5:
            self.logger.warning(f"Safety Block: {critique.flaws}")
            return

        # 2. Autonomy: Permission Gate (Simplified Mapping)
        # Determine risk heuristically or via critique
        risk = ActionRisk.HIGH if "restart" in command or "delete" in command else ActionRisk.LOW
        
        if not await self.autonomy.request_action("execute_command", risk, command):
            self.logger.warning("Autonomy Block: Permission Denied.")
            return

        # 3. Assembly: Construct Reasoning Pipeline
        self.logger.info("[2] Assembly: Building Model Pipeline...")
        pipeline = [
            PipelineStep(name="Plan", model_selector="small", system_prompt="Create a plan."),
            PipelineStep(name="Execute", model_selector="huge", system_prompt="execution logic")
        ]
        report = await self.assembly.run_pipeline(pipeline, command)
        
        self.logger.info(f"[3] Execution Result: {report.final_output}")
        
        # 4. Memory: Log Success
        await self.memory.add_memory(MemoryType.EPHEMERAL, f"Executed: {command}", {})

    # --- Simulation & Prediction ---

    async def simulate_scenario(self, tasks_count: int):
        self.logger.info(f"\n[SIMULATION] What-If we add {tasks_count} tasks?")
        snapshot = self.world.create_snapshot([{ "id": "thor", "cpu_cores": 32, "ram_gb": 128, "vram_gb": 24 }])
        
        new_tasks = [SimulatedTask(id=f"sim-{i}", cpu_cores=4, ram_gb=16, gpu_load=0, vram_gb=0, duration_seconds=10) for i in range(tasks_count)]
        snapshot.pending_tasks = new_tasks
        
        result = self.world.run_scenario(snapshot, 30)
        self.logger.info(f"Outcome: {len(result.completed_tasks)} completed, {len(result.failed_tasks)} failed.")

    async def analyze_cluster_future(self):
        self.logger.info("\n[PREDICTION] Forecasting Node Health...")
        # Inject dummy telemetry
        self.forecasting.ingest_telemetry("thor", TelemetryPoint(timestamp=time.time(), cpu_usage=50, gpu_usage=90, vram_usage_gb=20, temp_c=85, tasks_running=5))
        
        forecast = await self.forecasting.predict_node_state("thor", horizon_seconds=300)
        self.logger.info(f"Forecast for Thor: Failed Prob={forecast.failure_probability}")
        if forecast.failure_probability > 0.6:
            self.logger.warning("Recommendation: " + forecast.reasoning)

    # --- Automation & Learning ---

    async def trigger_automation_chains(self):
        self.logger.info("\n[AUTOMATION] Checking Logic Chains...")
        # Mock metrics
        metrics = {"gpu": 95, "ram": 40}
        events = ["HEARTBEAT"]
        
        # Add a test chain
        self.automation.load_chains([{
            "name": "GPU Safety",
            "triggers": [{"type": "threshold", "condition": "gpu > 90"}],
            "actions": [{"type": "log", "payload": {"msg": "GPU Heat Warning!"}}]
        }])
        
        await self.automation.evaluate_state(metrics, events)

    async def learn_and_improve(self):
        self.logger.info("\n[LEARNING] Nightly Skill Synthesis...")
        # Seed memory
        await self.memory.add_memory(MemoryType.LONG_TERM, "Reflection: Task X failed due to VRAM OOM.", {})
        
        report = await self.skills.run_nightly_learning()
        if report.new_skills:
            self.logger.info(f"Learned New Skill: {report.new_skills[0].name}")

# --- Demo: Day in the Life of STAN ---

async def run_supernova_demo():
    print("=== STAN SUPERNOVA AI: ONLINE ===")
    
    # 1. Setup Mock AI to handle all these diverse calls
    from triforce.odin.stan.ai_provider import AIProvider
    class SuperMock(AIProvider):
        async def generate(self, prompt, system, **kwargs):
            # Router for different components
            if "CRITIQUE" in system: 
                return json.dumps({"score": 0.9, "flaws": [], "suggestion": "Safe."}) # Metacognition
            if "LEARNING" in system:
                return json.dumps({"skills": [{"name": "VRAM Management", "category": "optimization", "trigger_condition": "VRAM > 90%", "action_logic": "Throttle", "confidence": 0.9}]})
            if "PRECOGNITION" in system:
                return json.dumps({"failure_prob": 0.75, "risks": ["Overheat"], "recommendation": "Cool down"})
            return "Generic Success Response."
            
        async def embed(self, t): return [0.0]*128
        async def classify(self, t, l): return l[0]

    stan = STANSupernova()
    # Inject patch
    stan.ai = SuperMock(ProviderConfig(type="mock"))
    # Propagate mock to sub-systems
    stan.memory.ai = stan.ai
    stan.forecasting.stats = stan.forecasting.stats # Keep stats
    stan.forecasting.semantic.ai = stan.ai
    stan.skills.ai = stan.ai
    stan.metacognition.ai = stan.ai
    stan.assembly.ai = stan.ai
    
    # --- Step 1: Receiving a Command ---
    await stan.act("Deploy Llama-3-70B to Thor")
    
    # --- Step 2: Predicting Failures ---
    await stan.analyze_cluster_future()
    
    # --- Step 3: Simulation ---
    await stan.simulate_scenario(tasks_count=10) # Overload test
    
    # --- Step 4: Reactive Automation ---
    await stan.trigger_automation_chains()
    
    # --- Step 5: Learning ---
    await stan.learn_and_improve()
    
    print("\n=== STAN SUPERNOVA AI: STANDBY ===")

if __name__ == "__main__":
    asyncio.run(run_supernova_demo())
