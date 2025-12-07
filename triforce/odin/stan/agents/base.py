import asyncio
import logging
import uuid
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

# Imports
from triforce.odin.stan.ai_provider import AIProvider
from triforce.odin.stan.memory import MemoryEngine, MemoryType

class AgentBase(ABC):
    """
    Abstract Base Class for Autonomous STAN Agents.
    """
    
    SYSTEM_PROMPT_TEMPLATE = """
    You are {agent_name}, a specialized sub-agent of STAN.
    ROLE: {role_description}
    CURRENT GOAL: {goal}
    
    You have access to RAG memory and cluster telemetry.
    Act autonomously to fulfill your goal.
    """

    def __init__(self, 
                 stan_core: Any, # Reference to parent STAN for reporting/tools
                 role: str, 
                 goal: str,
                 ai_provider: AIProvider,
                 memory: MemoryEngine):
        
        self.agent_id = f"{role.lower()}-{uuid.uuid4().hex[:6]}"
        self.role = role
        self.goal = goal
        self.stan = stan_core
        self.ai = ai_provider
        self.memory = memory
        
        self.logger = logging.getLogger(f"stan.agent.{self.agent_id}")
        self._running = False
        self._shutdown_event = asyncio.Event()
        self.inbox = asyncio.Queue()

    async def start(self):
        """Lifecycle: Start"""
        self.logger.info(f"Agent {self.agent_id} ({self.role}) starting...")
        self._running = True
        try:
            await self.run_loop()
        except asyncio.CancelledError:
            self.logger.info("Agent cancelled.")
        except Exception as e:
            self.logger.error(f"Agent crashed: {e}")
            await self._report_failure(str(e))
        finally:
            self._running = False
            self.logger.info(f"Agent {self.agent_id} stopped.")

    async def stop(self):
        """Lifecycle: Stop"""
        self._running = False
        self._shutdown_event.set()

    @abstractmethod
    async def run_loop(self):
        """
        Main specific logic for the agent.
        Must check self._running periodically.
        """
        pass

    async def think(self, context: str) -> Dict[str, Any]:
        """
        Generic AI reasoning step using the agent's persona.
        """
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.role,
            role_description=self.__doc__ or "Helper Agent",
            goal=self.goal
        )
        
        full_context = await self.memory.get_context_for_reasoning(context)
        
        try:
            response = await self.ai.generate(
                prompt=f"Context: {full_context}\nTask: {context}",
                system=system_prompt,
                json_format=True
            )
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Thinking failed: {e}")
            return {"error": str(e)}

    async def send_message(self, recipient_id: str, content: Any):
        """
        Intra-swarm communication.
        """
        # In a real system, this would look up the agent in a registry.
        # For now, we assume STAN routes it or we mock it.
        self.logger.info(f"Sending MSG to {recipient_id}: {content}")
        # self.stan.route_message(recipient_id, content) # Hypothetical

    async def _report_failure(self, error: str):
        self.logger.error(f"Reporting failure to STAN: {error}")
        await self.stan.memory.ingest_log({
            "level": "ERROR",
            "message": f"Agent {self.agent_id} Failed: {error}"
        })

    def __repr__(self):
        return f"<Agent {self.agent_id}: {self.role}>"
