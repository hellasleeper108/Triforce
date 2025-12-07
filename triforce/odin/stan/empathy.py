import asyncio
import logging
import json
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from triforce.odin.stan.ai_provider import AIProvider

# --- Data Models ---

class UserEmotion(str, Enum):
    FOCUSED = "focused"
    FRUSTRATED = "frustrated"
    OVERWHELMED = "overwhelmed"
    PLAYFUL = "playful"
    URGENT = "urgent"
    TIRED = "tired"
    NEUTRAL = "neutral"

class UserState(BaseModel):
    emotion: UserEmotion
    confidence: float
    intent_summary: str
    evidence: List[str] # Key words or phrases triggering this

class ResponseStyle(BaseModel):
    verbosity: float = Field(..., description="0.0 (Silent) to 1.0 (Verbose)")
    formality: float = Field(..., description="0.0 (Casual) to 1.0 (Formal)")
    empathy_level: float = Field(..., description="0.0 (Robotic) to 1.0 (Supportive)")
    suggest_actions: bool = True

# --- Empathy Engine ---

class EmpathyEngine:
    """
    The EQ processing unit for STAN.
    Adapts STAN's personality based on User State.
    """
    
    SYSTEM_PROMPT = """
    Analyze the user's input for Emotional State and Intent.
    
    States:
    - FOCUSED: Clear, technical commands. 
    - FRUSTRATED: Complaints, swearing, repetition, "why".
    - URGENT: "Now", "Immediate", "Emergency", capitalization.
    - PLAYFUL: Jokes, slang, hypothetical scenarios.
    - TIRED: Typos, short vague queries.
    - OVERWHELMED: "Help", "Too much", confusion.
    
    Output JSON:
    {
      "emotion": "...",
      "confidence": 0.0-1.0,
      "intent_summary": "...",
      "evidence": ["..."]
    }
    """

    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider
        self.logger = logging.getLogger("stan.empathy")

    async def infer_user_state(self, text: str, voice_metadata: Dict[str, Any] = None) -> UserState:
        """
        Derive state from text + optional voice cues (pitch, pace).
        """
        # Mix in metadata if available
        context = f"Input: \"{text}\"\n"
        if voice_metadata:
            context += f"Voice Meta: {voice_metadata}\n"

        try:
            response = await self.ai.generate(
                prompt=context,
                system=self.SYSTEM_PROMPT,
                json_format=True
            )
            data = json.loads(response)
            return UserState(**data)
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            # Fallback
            return UserState(emotion=UserEmotion.NEUTRAL, confidence=0.0, intent_summary="Unknown", evidence=[])

    def suggest_response_style(self, state: UserState) -> ResponseStyle:
        """
        Maps emotional state to optimal communication strategy.
        """
        e = state.emotion
        
        if e == UserEmotion.FRUSTRATED:
            # Be quick, apologetic, and solution-oriented. Don't yap.
            return ResponseStyle(verbosity=0.2, formality=0.8, empathy_level=0.9, suggest_actions=True)
            
        if e == UserEmotion.URGENT:
            # Military precision. No fluff.
            return ResponseStyle(verbosity=0.1, formality=0.5, empathy_level=0.0, suggest_actions=False)
            
        if e == UserEmotion.PLAYFUL:
            # Match energy. Be verbose and witty.
            return ResponseStyle(verbosity=0.9, formality=0.1, empathy_level=0.5, suggest_actions=True)
            
        if e == UserEmotion.OVERWHELMED:
            # Guide gently. Step-by-step.
            return ResponseStyle(verbosity=0.6, formality=0.6, empathy_level=1.0, suggest_actions=True)
            
        if e == UserEmotion.FOCUSED:
            # Standard professional.
            return ResponseStyle(verbosity=0.5, formality=0.7, empathy_level=0.2, suggest_actions=True)

        return ResponseStyle(verbosity=0.5, formality=0.5, empathy_level=0.5, suggest_actions=True)

# --- Demo Driver ---

async def run_empathy_demo():
    from triforce.odin.stan.ai_provider import ProviderConfig
    
    # Mock AI
    class EmpathyMock(AIProvider):
        async def generate(self, prompt, system, **kwargs):
            p_lower = prompt.lower()
            
            if "damn" in p_lower or "why" in p_lower:
                return json.dumps({
                    "emotion": "frustrated",
                    "confidence": 0.95,
                    "intent_summary": "User is angry about a failure.",
                    "evidence": ["damn it", "why"]
                })
            
            if "lol" in p_lower or "cool" in p_lower:
                return json.dumps({
                    "emotion": "playful",
                    "confidence": 0.8,
                    "intent_summary": "User is experimenting or having fun.",
                    "evidence": ["lol", "cool"]
                })
                
            if "emergency" in p_lower:
                return json.dumps({
                    "emotion": "urgent",
                    "confidence": 0.99,
                    "intent_summary": "Immediate action required.",
                    "evidence": ["EMERGENCY"]
                })

            return json.dumps({
                "emotion": "focused",
                "confidence": 0.7,
                "intent_summary": "Standard command.",
                "evidence": []
            })
            
        async def embed(self, t): return []
        async def classify(self, t, l): return l[0]

    ai = EmpathyMock(ProviderConfig(type="mock"))
    engine = EmpathyEngine(ai)
    
    examples = [
        "Why the hell does this keep failing?! Damn it.",
        "lol check out this huge model I found, let's allow it.",
        "SYSTEM EMERGENCY. SHUT DOWN EVERYTHING.",
        "Deploy the containment pod to node 4."
    ]
    
    print("--- User Empathy Analysis ---")
    for text in examples:
        state = await engine.infer_user_state(text)
        style = engine.suggest_response_style(state)
        
        print(f"\nInput: \"{text}\"")
        print(f"State: [{state.emotion.name}] (Conf: {state.confidence})")
        print(f"Style: Verbosity={style.verbosity}, Formality={style.formality}, Empathy={style.empathy_level}")

if __name__ == "__main__":
    asyncio.run(run_empathy_demo())
