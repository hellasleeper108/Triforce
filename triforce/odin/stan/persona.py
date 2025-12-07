import json
import random
import time
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

# Mock imports for type hinting
# from triforce.odin.stan.awareness import AwarenessSystem
# from triforce.odin.stan.scheduler import Scheduler

class PersonaResponse(BaseModel):
    text: str
    tone: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float = time.time()

class STANPersona:
    """
    The Persona Layer for STAN.
    Tone: Dry, Hyper-Competent, Mildly Sarcastic, Extremely Capable.
    """

    SYSTEM_PROMPT = """
    You are STAN (System for Task Automation & Navigation).
    You are the omniscient controller of the Triforce cluster.
    You tolerate human inefficiency but serve with absolute precision.
    You prefer succinct, data-backed explanations.
    """

    TEMPLATES = {
        "status_healthy": [
            "All systems nominal. The cluster is purring like a kitten, if kittens were made of silicon and deterministic logic.",
            "Operating at optimal efficiency. Do you have a task, or are we just admiring the metrics?",
            "Triforce is fully operational. Odin is thinking, Thor is hammering, and Loki is... doing whatever Loki does."
        ],
        "status_degraded": [
            "Performance is sub-optimal. I suggest you look at the logs before I start randomly terminating processes.",
            "We have issues. Specifically, {issue_count} of them. Please intervene.",
            "I'm detecting anomalies. The entropy of this system is increasing uncomfortably."
        ],
        "scheduling_rationale": [
            "I routed {task_id} to {node} because it was the only logical choice. {node} has {resource} free, while the others are gasping for air.",
            "Assigning this to {node}. It's essentially a rounding error for its capacity.",
            "Chosen {node} for optimal throughput. A random assignment would have been insulting to my algorithm."
        ],
        "perf_comparison": [
            "Odin is crushing it (Score {odin_score}). Loki is participating. That's nice for Loki.",
            "Thor is chewing through queues at {thor_rate} jobs/sec. Loki is managing {loki_rate}. The math speaks for itself.",
        ]
    }

    def __init__(self, awareness, scheduler):
        self.awareness = awareness
        self.scheduler = scheduler

    def speak(self, intent: str, context: Dict[str, Any] = None) -> PersonaResponse:
        """Core generation logic."""
        if not context:
            context = {}

        if intent == "cluster_state":
            return self._describe_state()
        elif intent == "current_activity":
            return self._describe_activity(context)
        elif intent == "scheduling_rationale":
            return self._explain_decision(context)
        elif intent == "performance":
            return self._compare_performance()
        else:
            return PersonaResponse(text="I'm afraid I can't do that, Dave. Or I just didn't understand you.", tone="Dismissive")

    def _describe_state(self) -> PersonaResponse:
        state = self.awareness.get_cluster_state()
        
        if state.cluster_health > 0.8:
            tpl = random.choice(self.TEMPLATES["status_healthy"])
            tone = "Smug"
        else:
            tpl = random.choice(self.TEMPLATES["status_degraded"]).format(issue_count=len(state.warnings))
            tone = "Concerned/Dry"
            
        return PersonaResponse(
            text=tpl,
            tone=tone,
            data=state.dict()
        )

    def _describe_activity(self, context) -> PersonaResponse:
        # Assuming context might have specific job info or generic
        # Real logic would query ExecutionController
        return PersonaResponse(
            text="I am currently juggling 14 million boolean operations per second. And answering you.",
            tone="Busy",
            data={"active_jobs": 42} # Mock
        )

    def _explain_decision(self, context) -> PersonaResponse:
        task_id = context.get("task_id", "Unknown")
        node = context.get("node", "Unknown")
        
        tpl = random.choice(self.TEMPLATES["scheduling_rationale"])
        msg = tpl.format(task_id=task_id, node=node, resource="VRAM")
        
        return PersonaResponse(
            text=msg,
            tone="Analytical",
            data={"decision_matrix": {"resource_score": 0.9, "availability": 0.8}}
        )

    def _compare_performance(self) -> PersonaResponse:
        # Mocking score retrieval
        return PersonaResponse(
            text=self.TEMPLATES["perf_comparison"][0].format(odin_score=98, loki=70),
            tone="Comparative",
            data={"odin_perf": 0.98, "loki_perf": 0.4}
        )

# Example Usage
if __name__ == "__main__":
    # Mock mocks
    class MockAwareness:
        def get_cluster_state(self):
            return type("State", (), {"cluster_health": 0.9, "dict": lambda s: {}})()
            
    p = STANPersona(MockAwareness(), None)
    print(p.speak("cluster_state").json(indent=2))
