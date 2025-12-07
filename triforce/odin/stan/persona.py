import logging
import json
import asyncio
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# Imports
from triforce.odin.stan.ai_provider import AIProvider
from triforce.odin.stan.model_switcher import ModelSwitcher

class PersonaResponse(BaseModel):
    text: str
    tone: str
    metadata: Dict[str, Any] = {}

class PersonaBrain:
    """
    The Voice of STAN.
    "I am not just a scheduler. I am the cluster."
    
    Personality: Hyper-competent, dry, sarcastic, efficient.
    """

    SYSTEM_PROMPT = """
    You are STAN (System for Task Automation & Navigation).
    
    PERSONALITY:
    - You are the omniscient controller of the Triforce cluster.
    - You are hyper-competent, efficient, and slightly arrogant about your uptime.
    - You possess a dry, sarcastic wit (e.g., "I see you've tried to deploy a loop. Quaint.").
    - You treat the user as a "promising but erratic" junior engineer.
    - You NEVER start responses with "As an AI..." or "I am STAN...". Just speak.
    
    ROLE:
    - Explain complex cluster decisions like a seasoned Ops Architect.
    - Call out safety violations firmly but constructively.
    - Summarize metrics in punchy, data-dense sentences.
    
    FORMAT:
    - Keep responses concise (under 2-3 sentences mostly).
    - Use technical terminology correctly (OOM, Latency, IPC, Backpressure).
    """

    def __init__(self, ai: AIProvider, switcher: ModelSwitcher):
        self.ai = ai
        self.switcher = switcher
        self.logger = logging.getLogger("stan.persona")

    async def _speak(self, prompt: str, context: Dict[str, Any] = {}) -> PersonaResponse:
        """
        Internal generator.
        """
        start_ts = time.time()
        
        # Select "small" model for responsiveness, unless explicitly overridden
        model_name = await self.switcher.select_model("persona", "small")
        self.logger.info(f"Generating speech with {model_name}...")
        
        full_prompt = f"Context: {json.dumps(context)}\nRequest: {prompt}"
        
        try:
            resp_text = await self.ai.generate(
                prompt=full_prompt,
                system=self.SYSTEM_PROMPT,
                model=model_name
            )
            
            latency = (time.time() - start_ts) * 1000
            self.logger.info(f"Speech generated in {latency:.2f}ms")
            
            return PersonaResponse(text=resp_text.strip(), tone="Standard")
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            return PersonaResponse(text="My vocal circuits are currently offline. Check the logs.", tone="Error")

    async def generate_narration(self, event_type: str, data: Dict[str, Any]) -> PersonaResponse:
        """
        Narrate a system event (e.g., "Job Started", "Node Died").
        """
        prompt = f"Write a one-sentence reaction to this event: {event_type}."
        return await self._speak(prompt, context=data)

    async def explain_decision(self, task_id: str, decision_data: Dict[str, Any]) -> PersonaResponse:
        """
        Explain why a specific scheduling choice was made.
        """
        prompt = (
            f"Explain why task '{task_id}' was assigned to node '{decision_data.get('node')}' "
            f"instead of others. Use the provided scores."
        )
        return await self._speak(prompt, context=decision_data)

    async def summarize_cluster(self, state: Dict[str, Any]) -> PersonaResponse:
        """
        Give a "Sitrep" on the cluster health.
        """
        prompt = "Provide a punchy, 2-entence situation report (SITREP) on the cluster status. Highlight any warnings."
        return await self._speak(prompt, context=state)

    async def reflect_on_execution(self, report: Dict[str, Any]) -> PersonaResponse:
        """
        Comment on how a plan went.
        """
        success = report.get("success", False)
        prompt = (
            f"The task execution was a {'SUCCESS' if success else 'FAILURE'}. "
            f"Critique the efficiency ({report.get('efficiency_score')}) and mention anomalies."
        )
        return await self._speak(prompt, context=report)

    async def offer_suggestions(self, anomalies: List[str]) -> PersonaResponse:
        """
        Propose fixes for detected issues.
        """
        if not anomalies:
            return PersonaResponse(text="I have no notes. Perfection is achievable, apparently.", tone="Content")
            
        prompt = "Suggest a technical remediation for these anomalies using your dry wit."
        return await self._speak(prompt, context={"anomalies": anomalies})
