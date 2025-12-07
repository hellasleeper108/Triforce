import asyncio
import logging
import uuid
import time
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

# --- Integration ---
from triforce.odin.stan.supernova import STANSupernova
from triforce.odin.stan.empathy import EmpathyEngine, UserState, ResponseStyle

# --- Data Models ---

class CommandSource(str, Enum):
    CLI = "cli"
    WEB = "web"
    VOICE = "voice"
    API = "api"

class UserCommand(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: CommandSource
    raw_text: str
    cleaned_text: str
    timestamp: float = Field(default_factory=time.time)
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class ResponseType(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    RESULT = "result"
    EXPLANATION = "explanation"

class STANResponse(BaseModel):
    command_id: str
    message_text: str
    message_type: ResponseType
    structured_data: Optional[Dict[str, Any]] = None
    actions_taken: List[str] = []
    severity_level: str = "normal" # normal, high, critical
    voice_synthesis_text: Optional[str] = None # Text optimized for TTS

# --- UX Orchestrator ---

class UXOrchestrator:
    """
    Standardizes interaction flow:
    User -> [Command] -> Empathy -> Supernova -> Persona -> [Response] -> User
    """
    
    def __init__(self):
        self.logger = logging.getLogger("stan.ux")
        self.supernova = STANSupernova(use_mock=True)
        self.empathy = EmpathyEngine(self.supernova.ai)

    async def process_user_command(self, cmd: UserCommand) -> STANResponse:
        self.logger.info(f"[{cmd.source.upper()}] Processing: '{cmd.cleaned_text}'")
        
        # 1. Infer User State
        user_state = await self.empathy.infer_user_state(cmd.cleaned_text, cmd.metadata)
        style = self.empathy.suggest_response_style(user_state)
        
        self.logger.info(f"User State: {user_state.emotion.name} (Verbosity: {style.verbosity})")

        # 2. Execute Logic (Supernova)
        # Note: Supernova.act() is currently void/logging-only in our mock.
        # In a real system, we'd capture the execution report.
        try:
            await self.supernova.act(cmd.cleaned_text)
            actions = ["Analyzed Intent", "Executed Plan", "Verified Outcome"]
            status = "success"
        except Exception as e:
            self.logger.error(f"Execution Error: {e}")
            actions = ["Attempted Execution", "Failed"]
            status = "error"

        # 3. Generate Persona-Aware Response
        # (Simulating PersonaBrain generation based on style)
        response_text = self._generate_response_text(cmd.cleaned_text, user_state, style, status)
        
        # 4. Construct Canonical Response
        response = STANResponse(
            command_id=cmd.id,
            message_text=response_text,
            message_type=ResponseType.RESULT if status == "success" else ResponseType.ERROR,
            actions_taken=actions,
            severity_level="normal" if status == "success" else "high",
            voice_synthesis_text=response_text # Could be simplified for voice
        )
        
        return response

    def _generate_response_text(self, input_text: str, state: UserState, style: ResponseStyle, status: str) -> str:
        """
        Mock Persona Generation logic.
        """
        if status == "error":
            if style.empathy_level > 0.7:
                return "I'm having trouble with that request. I apologize. Let me check the logs."
            return "Execution Failed. Check system logs."

        # Success Logic
        if "status" in input_text.lower():
            if style.verbosity > 0.8: # Playful/Chatty
                return "All systems are green and looking good! We're running at 98% efficiency."
            if style.verbosity < 0.3: # Urgent/Frustrated
                return "Status: Nominal."
            return "Cluster status is nominal. All nodes online."

        if "deploy" in input_text.lower():
            if style.empathy_level > 0.8: # Overwhelmed user?
                return "I've started the deployment for you. Sit tight, I'll update you when it's done."
            return "Deployment initiated."
            
        return "Command executed successfully."

# --- Demo Driver ---

async def run_ux_demo():
    print("--- STAN UX Orchestration ---")
    orchestrator = UXOrchestrator()

    # Scenario 1: CLI User (Focused)
    cmd_cli = UserCommand(
        source=CommandSource.CLI,
        raw_text="stanctl status",
        cleaned_text="Check status",
        metadata={}
    )
    resp_cli = await orchestrator.process_user_command(cmd_cli)
    print(f"\n[CLI Response]: {resp_cli.message_text}")
    print(f"  > Actions: {resp_cli.actions_taken}")

    # Scenario 2: Web User (Frustrated)
    cmd_web = UserCommand(
        source=CommandSource.WEB,
        raw_text="Why isn't the deployment working?! Status now!",
        cleaned_text="Check status immediately",
        metadata={"pacing": "fast"}
    )
    # Mocking empathy engine to force FRUSTRATED for demo purposes if needed, 
    # but the empathy engine mock in `empathy.py` might handle "why" well.
    # Let's see if the mock provider in `empathy.py` is used.
    # UXOrchestrator inits EmpathyEngine with Supernova.ai.
    # Supernova.ai is default OpenAI or Mock. 
    # We need to make sure we inject a smart enough mock or use the one from Empathy.
    
    # Injecting custom mock behavior for consistency in this demo script
    from triforce.odin.stan.ai_provider import AIProvider
    class DemoMock(AIProvider):
        async def generate(self, prompt, system, **kwargs):
            if "why" in prompt.lower():
                return '{"emotion": "frustrated", "confidence": 0.9, "intent_summary": "Anger", "evidence": []}'
            return '{"emotion": "focused", "confidence": 0.9, "intent_summary": "Inquiry", "evidence": []}'
        async def embed(self, t): return []
        async def classify(self, t, l): return l[0]

    from triforce.odin.stan.ai_provider import ProviderConfig
    orchestrator.empathy.ai = DemoMock(ProviderConfig(type="mock"))

    resp_web = await orchestrator.process_user_command(cmd_web)
    print(f"\n[Web Response]: {resp_web.message_text}")
    print(f"  > Style Adapted: (User was {orchestrator.empathy.suggest_response_style(await orchestrator.empathy.infer_user_state(cmd_web.cleaned_text)).verbosity} verbosity)")

if __name__ == "__main__":
    asyncio.run(run_ux_demo())
