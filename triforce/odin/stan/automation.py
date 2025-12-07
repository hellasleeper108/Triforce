import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel

# --- Primitives ---

class TriggerType(str, Enum):
    THRESHOLD = "threshold"
    EVENT = "event"
    SCHEDULE = "schedule"

class ActionType(str, Enum):
    LOG = "log"
    NOTIFY = "notify"
    MIGRATE = "migrate"
    SPAWN_AGENT = "spawn_agent"
    REPLAN = "replan"

class TriggerConfig(BaseModel):
    type: TriggerType
    condition: str  # "gpu > 90", "event == TASK_FAIL", "cron 0 0 * * *"
    target_node: Optional[str] = None

class ActionConfig(BaseModel):
    type: ActionType
    payload: Dict[str, Any]

class AutomationChain(BaseModel):
    name: str
    enabled: bool = True
    triggers: List[TriggerConfig]
    actions: List[ActionConfig]

# --- Engine ---

class AutomationEngine:
    """
    Reactive Workflow Engine.
    "If This, Then That" for the Cluster.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("stan.automation")
        self.chains: List[AutomationChain] = []
        self._running = False

    def load_chains(self, chains_data: List[Dict[str, Any]]):
        self.chains = [AutomationChain(**c) for c in chains_data]
        self.logger.info(f"Loaded {len(self.chains)} automation chains.")

    async def evaluate_state(self, cluster_metrics: Dict[str, Any], event_stream: List[str]):
        """
        Main Loop Step: Check all chains against current state.
        """
        for chain in self.chains:
            if not chain.enabled:
                continue
                
            triggered = False
            for trig in chain.triggers:
                if self._check_trigger(trig, cluster_metrics, event_stream):
                    triggered = True
                    break # One trigger is enough (OR logic)
            
            if triggered:
                self.logger.info(f"CHAIN TRIGGERED: {chain.name}")
                await self._execute_actions(chain)
    
    def _check_trigger(self, trigger: TriggerConfig, metrics: Dict[str, Any], events: List[str]) -> bool:
        try:
            if trigger.type == TriggerType.THRESHOLD:
                # Naive eval (Safe for demo, unsafe for prod if user input)
                # Expects context variables: gpu, cpu, ram, active_tasks
                # Format: "gpu > 90" -> metrics['gpu'] > 90
                # We'll map metrics dict to locals
                # This is a simplifiction. Real engine uses AST parsing.
                return self._eval_safe(trigger.condition, metrics)

            if trigger.type == TriggerType.EVENT:
                # Check if condition string exists in recent events
                for e in events:
                    if trigger.condition in e:
                        return True
                return False

            if trigger.type == TriggerType.SCHEDULE:
                # Mock schedule hit
                return trigger.condition == "NOW"

        except Exception as e:
            self.logger.error(f"Trigger Check Failed '{trigger.condition}': {e}")
            return False

    def _eval_safe(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Simple safe evaluation of 'key > value'.
        """
        try:
            # Tokenize: "gpu", ">", "90"
            parts = condition.split()
            if len(parts) != 3: return False
            key, op, val = parts[0], parts[1], float(parts[2])
            
            current_val = context.get(key, 0)
            
            if op == ">": return current_val > val
            if op == "<": return current_val < val
            if op == "==": return current_val == val
            return False
        except ValueError:
            return False

    async def _execute_actions(self, chain: AutomationChain):
        for action in chain.actions:
            self.logger.info(f"  -> ACTION: {action.type} | {action.payload}")
            
            if action.type == ActionType.NOTIFY:
                # self.stan.notify_user(...)
                pass
            elif action.type == ActionType.SPAWN_AGENT:
                # self.stan.agents.spawn(...)
                pass
            # ... other handlers

# --- Demo Driver ---

async def run_automation_demo():
    logging.basicConfig(level=logging.INFO)
    print("--- 1. Loading Chains ---")
    
    # Define Chains (YAML-like structure)
    chain_defs = [
        {
            "name": "OOM Watchdog",
            "triggers": [
                {"type": "threshold", "condition": "ram > 90"}
            ],
            "actions": [
                {"type": "log", "payload": {"msg": "High RAM detected!"}},
                {"type": "notify", "payload": {"user": "admin", "msg": "Node critically low on RAM"}}
            ]
        },
        {
            "name": "Task Failure Handler",
            "triggers": [
                {"type": "event", "condition": "TASK_FAIL"}
            ],
            "actions": [
                {"type": "replan", "payload": {"strategy": "retry_on_different_node"}}
            ]
        },
        {
            "name": "Nightly Report",
            "triggers": [
                {"type": "schedule", "condition": "NOW"} # Mocked time trigger
            ],
            "actions": [
                {"type": "spawn_agent", "payload": {"agent_role": "Scribe", "task": "Generate Summary"}}
            ]
        }
    ]
    
    engine = AutomationEngine()
    engine.load_chains(chain_defs)
    
    print("\n--- 2. Simulating Cluster State ---")
    
    # Scenario A: Healthy Cluster
    print("Status: Healthy")
    metrics = {"ram": 40, "gpu": 20}
    events = ["TASK_START", "HEARTBEAT"]
    await engine.evaluate_state(metrics, events)
    
    # Scenario B: High Memory
    print("\nStatus: Memory Spike")
    metrics = {"ram": 95, "gpu": 50} 
    await engine.evaluate_state(metrics, events)
    
    # Scenario C: Event Trigger
    print("\nStatus: Task Crash")
    metrics = {"ram": 40, "gpu": 20}
    events = ["HEARTBEAT", "TASK_FAIL error=OOM"]
    await engine.evaluate_state(metrics, events)
    
    # Scenario D: Schedule
    print("\nStatus: Scheduled Time")
    # We patch the trigger logic for the demo to hit 'schedule'
    # In real code, this would check system time.
    # The 'condition="NOW"' is our mock flag here.

if __name__ == "__main__":
    asyncio.run(run_automation_demo())
