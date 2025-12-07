import asyncio
import json
import random
import time
from typing import List, Dict, Any

from triforce.odin.stan.agents.base import AgentBase

# --- 1. ShardMaster Agent ---

class ShardMasterAgent(AgentBase):
    """
    Role: Decompose large tasks into parallel shards.
    Capabilities: JSON splitting, node assignment strategy.
    """
    
    async def run_loop(self):
        while self._running:
            # Check for tasks in queue (simulated)
            try:
                task = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                await self.process_task(task)
            except asyncio.TimeoutError:
                await asyncio.sleep(0.5)

    async def process_task(self, task: Dict[str, Any]):
        self.logger.info(f"Sharding task: {task.get('id')}")
        
        # AI Reasoning for Sharding Strategy
        decision = await self.think(f"Determine optimal sharding for: {json.dumps(task)}")
        num_shards = decision.get("shards", 2)
        
        self.logger.info(f"Split into {num_shards} shards. Dispatching...")
        # Simulate dispatch
        for i in range(num_shards):
            self.logger.info(f" -> Dispatching Shard {i+1}/{num_shards}")
            await asyncio.sleep(0.1)
            
        await self.stan.memory.ingest_log({
            "level": "INFO", 
            "message": f"ShardMaster decomposed {task.get('id')} into {num_shards} chunks."
        })

# --- 2. Scribe Agent ---

class ScribeAgent(AgentBase):
    """
    Role: System Historian.
    Capabilities: Summarize logs, write reports, update Knowledge Base.
    """

    async def run_loop(self):
        while self._running:
            # Periodically wake up to summarize recent history
            await asyncio.sleep(10) # 10s interval for demo
            await self.summarize_recent_activity()

    async def summarize_recent_activity(self):
        self.logger.info("Scribe waking up to summarize...")
        # Mock fetching recent logs
        summary = await self.think("Summarize the last 10 system logs.")
        
        if summary.get("summary"):
            self.logger.info(f"Drafted Report: {summary['summary']}")
            # Commit to Long Term Memory
            from triforce.odin.stan.memory import MemoryType
            await self.memory.add_memory(
                MemoryType.LONG_TERM,
                f"Scribe Report: {summary['summary']}",
                {"agent": self.agent_id}
            )

# --- 3. Optimizer Agent ---

class OptimizerAgent(AgentBase):
    """
    Role: Resource Tuner.
    Capabilities: Adjust batch sizes, timeouts, and cache settings.
    """
    
    async def run_loop(self):
        while self._running:
            # Check cluster health
            state = await self.stan.awareness.get_cluster_state()
            if state.avg_load > 80:
                await self.tune_resources(state)
            await asyncio.sleep(5)

    async def tune_resources(self, state):
        self.logger.warning(f"High Load Detected ({state.avg_load}%). Optimizing...")
        recommendation = await self.think(f"Suggest tuning for high load: {state.avg_load}%")
        
        action = recommendation.get("action", "reduce_batch_size")
        self.logger.info(f"Applying Optimization: {action}")
        # self.stan.config.update(action) # Hypothetical

# --- 4. Sentinel Agent ---

class SentinelAgent(AgentBase):
    """
    Role: Anomaly Watchdog.
    Capabilities: Deep packet inspection (telemetry), threat detection.
    """
    
    async def run_loop(self):
        while self._running:
            anomalies = await self.stan.awareness.detect_anomalies()
            if anomalies:
                await self.escalate_alert(anomalies)
            await asyncio.sleep(3)

    async def escalate_alert(self, start_anoms):
        # AI Filter: Is this real or noise?
        analysis = await self.think(f"Analyze anomalies: {json.dumps(start_anoms, default=str)}")
        
        if analysis.get("severity", "low") == "high":
            self.logger.critical(f"SENTINEL ALERT: {analysis.get('reason')}")
            # Notify STAN Persona to scream
            await self.stan.persona_brain.generate_narration("SECURITY ALERT", {"details": analysis})


# --- Example Swarm Driver ---

async def run_swarm_demo():
    from triforce.odin.stan.ai_core import STANAI
    from triforce.odin.stan.ai_provider import AIProvider, ProviderConfig, ModelConfig

    # 1. Setup STAN (Host Environment)
    # Using Mock with "Smart" responses for agents
    class SwarmMockProvider(AIProvider):
        async def generate(self, prompt, system, **kwargs):
            if "ShardMaster" in system:
                return json.dumps({"shards": 4, "strategy": "round_robin"})
            if "Scribe" in system:
                return json.dumps({"summary": "System ran 4 tasks successfully. No errors."})
            if "Optimizer" in system:
                return json.dumps({"action": "throttle_non_critical"})
            if "Sentinel" in system:
                return json.dumps({"severity": "high", "reason": "Unauthorized access attempt detected"})
            return "{}"
            
        async def embed(self, t): return [0.1]*128
        
        async def classify(self, text, labels): return labels[0]

    stan = STANAI(persistence_path=":memory:", use_mock=True)
    stan.ai = SwarmMockProvider(ProviderConfig(type="mock"))
    # Patch sub-components to use new AI
    stan.memory.ai = stan.ai
    stan.awareness.ai = stan.ai

    print("=== STAN Agent Swarm Demo ===\n")

    # 2. Spawn Agents
    agents = [
        ShardMasterAgent(stan, "ShardMaster", "Distribute compute", stan.ai, stan.memory),
        ScribeAgent(stan, "Scribe", "Log history", stan.ai, stan.memory),
        OptimizerAgent(stan, "Optimizer", "Maximize efficiency", stan.ai, stan.memory),
        SentinelAgent(stan, "Sentinel", "Guard cluster", stan.ai, stan.memory)
    ]

    # 3. Start Swarm
    tasks = [asyncio.create_task(a.start()) for a in agents]
    
    # 4. Simulate Activity
    print("[*] Agents Active...")
    
    # A. ShardMaster Work
    shard_master = agents[0]
    await shard_master.inbox.put({"id": "job-huge-1", "size_gb": 50})
    await asyncio.sleep(1)
    
    # B. Trigger Sentinel (Mock High Load/Anomaly via Awareness?)
    # We'll just wait for the loop to cycle once or twice.
    # Sentinel relies on Stan.awareness returning anomalies.
    # Let's mock awareness locally for this test context.
    async def mock_detect(): return {"node-1": "metric-spike"}
    stan.awareness.detect_anomalies = mock_detect
    
    await asyncio.sleep(4)
    
    # 5. Shutdown
    print("\n[*] Stopping Swarm...")
    for a in agents:
        await a.stop()
    
    # Wait for completion (they check _running and exit loops)
    # We just cancel the tasks to speed up demo exit if they are sleeping
    for t in tasks: t.cancel()
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(run_swarm_demo())
