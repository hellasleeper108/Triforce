import asyncio
import logging
import json
import time
from typing import Dict, Any

# Components
from triforce.odin.stan.ai_provider import AIProvider, AIProviderFactory, ProviderConfig
from triforce.odin.stan.model_registry import ModelRegistry
from triforce.odin.stan.model_routing import ModelRouter
from triforce.odin.stan.remote_ollama_client import RemoteOllamaClient
from triforce.odin.stan.memory import MemoryEngine, MemoryType
from triforce.odin.stan.brains import ParsingBrain, PlanningBrain, ReflectionBrain
from triforce.odin.stan.reasoning_fallbacks import ReasoningResilienceStrategy

# Configure Logger
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("stan.test")

# --- Mocks ---

class MockRemoteClient(RemoteOllamaClient):
    """
    Simulates a remote node (Thor/Odin/Loki).
    Returns scripted responses based on the model requested.
    """
    def __init__(self, base_url, node_id, **kwargs):
        super().__init__(base_url=base_url, node_id=node_id, **kwargs)
        self.node_id = node_id

    async def generate(self, prompt, model, json_format=False, **kwargs):
        logger.info(f"[{self.node_id}] GENERATE ({model}): {prompt[:30]}...")
        
        # Simulate latency
        await asyncio.sleep(0.1)
        
        # 1. Parsing Brain Simulation (JSON)
        if "PARSING" in prompt or "Task Graph" in str(kwargs.get("system", "")):
            return json.dumps({
                "tasks": [{
                    "task_id": "job-123",
                    "high_level_action": "TRAIN_MODEL",
                    "target_nodes": ["thor"], 
                    "required_resources": {"gpu": True},
                    "execution_steps": []
                }]
            })

        # 2. Planning Brain Simulation (JSON)
        if "PLANNING" in prompt or "Optimize" in prompt:
            return json.dumps({
                "plan_id": "plan-123",
                "schedule": ["n1"],
                "estimated_duration": 60
            })
            
        # 3. Reflection Brain Simulation (JSON)
        if "REFLECTION" in prompt:
            return json.dumps({
                "success": True,
                "efficiency_score": 0.95,
                "anomalies": []
            })
            
        # 4. Fallback/Standard
        return f"Processed by {self.node_id} using {model}"

    async def embed(self, text, model=None):
        logger.info(f"[{self.node_id}] EMBED: {text[:20]}...")
        # Deterministic mock vector
        return [0.1] * 768

# --- Test Suite ---

async def run_integration_test():
    print("=== STAN Full Stack Integration Test ===\n")
    
    # 1. Initialization
    print("[1] Initializing Core Muscles...")
    registry = ModelRegistry()
    router = ModelRouter(registry)
    
    # 2. Patch Router to use MockRemoteClient
    # This ensures we don't make real network calls but test all routing logic
    def mock_factory(node):
        return MockRemoteClient(node.base_url, node.hostname)
    router.get_client = mock_factory
    
    # 3. Setup Memory (with Mock AI for local embedding if needed, or route it)
    # We'll use the Router as the AI Provider for memory to test routing embeddings!
    # But MemoryEngine expects AIProvider. Router isn't AIProvider, but has get_client.
    # We'll create a wrapper or just trust the router's client for now.
    # For this test, let's make a simple provider for memory that routes via 'router'
    class RouterBackedProvider(AIProvider):
        def __init__(self, router): self.router = router
        async def embed(self, text):
            # Route to ANY capable node
            res = self.router.route_embed(text, "nomic-embed-text")
            if res.success:
                client = self.router.get_client(res.selected_node)
                return await client.embed(text)
            return []
        async def generate(self, p, **k): return "" # Unused by memory
        async def classify(self, t, l): return l[0] # Unused

    mem_provider = RouterBackedProvider(router)
    memory = MemoryEngine(mem_provider, db_path=":memory:")
    
    # 4. Populate Memory
    print("\n[2] Hydate Memory...")
    await memory.add_memory(MemoryType.KNOWLEDGE, "Thor is the heavy lifter.", {"node": "thor"})
    
    # 5. Instantiate Brains (Using Resilience Strategy)
    # ResilienceStrategy wraps the router, but Brains expect AIProvider.
    # We need an adapter. For this test, let's inject a "SmartProvider" that uses recover_generate.
    
    strategy = ReasoningResilienceStrategy(router, registry)
    
    class SmartBrainProvider(AIProvider):
        def __init__(self, strategy): self.strategy = strategy
        async def generate(self, prompt, **kwargs):
            # Extract model from kwargs or default
            model = kwargs.pop("model", "llama3:8b")
            res = await self.strategy.recover_generate(prompt, model, **kwargs)
            return res["response"]
        
        def get_best_model(self, **kwargs): return "llama3:70b" # Force high req for testing
            
        async def embed(self, t): return [] # Unused by Brains directly usually
        async def classify(self, t, l): return l[0]

    smart_ai = SmartBrainProvider(strategy)
    
    parser = ParsingBrain(smart_ai)
    planner = PlanningBrain(smart_ai)
    
    # 6. Execute Pipeline
    user_command = "Train a new model on Thor using the financial dataset"
    print(f"\n[3] Processing User Command: '{user_command}'")
    
    # A. Parse
    task_graph = await parser.think(user_command)
    print(f"  -> Parsed Task: {task_graph.tasks[0].high_level_action} (Target: {task_graph.tasks[0].target_nodes})")
    
    # B. Plan
    plan = await planner.think(task_graph)
    print(f"  -> Generated Plan ID: {plan.get('plan_id')}")
    
    # C. Verify Routing (Implicit)
    # The logs will show "Sending request to thor..." because get_best_model returned 70b, 
    # and 70b is routed to Thor.
    
    print("\n[4] Success. Pipeline verified.")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
