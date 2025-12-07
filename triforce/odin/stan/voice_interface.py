import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Optional

# Import STAN Supernova
from triforce.odin.stan.supernova import STANSupernova

# --- Abstractions ---

class SpeechToTextProvider(ABC):
    @abstractmethod
    async def listen(self) -> str:
        """Captures audio and returns transcribed text."""
        pass

class TextToSpeechProvider(ABC):
    @abstractmethod
    async def speak(self, text: str):
        """Synthesizes text to audio and plays it."""
        pass

# --- Mock Implementations ---

class MockSTT(SpeechToTextProvider):
    def __init__(self, inputs: list = None):
        self.inputs = inputs or []
        self._idx = 0

    async def listen(self) -> str:
        # Simulate listening delay
        await asyncio.sleep(1)
        if self._idx < len(self.inputs):
            text = self.inputs[self._idx]
            self._idx += 1
            print(f"[Mic] Captured: '{text}'")
            return text
        return "" # Silence

class MockTTS(TextToSpeechProvider):
    async def speak(self, text: str):
        # Simulate speaking delay
        print(f"[Speaker] STAN: \"{text}\"")
        await asyncio.sleep(len(text) * 0.05) 

# --- Voice Interface ---

class VoiceInterface:
    """
    The Verbal Interface for STAN.
    Orchestrates the Listen-Think-Speak loop.
    """
    
    def __init__(self, stt: SpeechToTextProvider, tts: TextToSpeechProvider):
        self.stt = stt
        self.tts = tts
        self.logger = logging.getLogger("stan.voice")
        # In a real app, this would connect to the running Supernova instance
        self.supernova = STANSupernova(use_mock=True) 

    async def run_loop(self):
        self.logger.info("Voice Interface Online. Waiting for wake word...")
        
        while True:
            # 1. Listen
            text = await self.stt.listen()
            if not text:
                break # End of mock input
                
            # 2. Wake Word Check (Simple string match for now)
            if "stan" in text.lower():
                await self.tts.speak("Listening.")
                
                # 3. Process Command
                clean_command = text.replace("Hey STAN", "").strip()
                self.logger.info(f"Processing command: {clean_command}")
                
                # We need to capture the text response from Supernova to speak it.
                # Since act() logs to stdout/logger, we'll wrap it or inspect the result.
                # For this demo, we'll simulate the response based on the command,
                # as act() currently doesn't return the text suitable for TTS directly.
                # In a full implementation, act() would return a structured response.
                
                # Hack: Simulate Supernova thinking
                await self.supernova.act(clean_command)
                
                # Generate a spoken response (Mocking what PersonaBrain would produce)
                response = f"I've initiated the protocol for {clean_command}. Cluster status is nominal."
                
                # 4. Speak Response
                await self.tts.speak(response)

# --- Demo Driver ---

async def run_voice_demo():
    print("--- STAN VOICE INTERFACE DEMO ---")
    
    # Pre-recorded inputs for the mock
    mock_inputs = [
        "Hey STAN, check system health.",
        "Hey STAN, deploy the new model to Thor.",
        "Ignore this background noise.", 
        "Hey STAN, shutdown sequence alpha."
    ]
    
    stt = MockSTT(inputs=mock_inputs)
    tts = MockTTS()
    
    voice = VoiceInterface(stt, tts)
    await voice.run_loop()

if __name__ == "__main__":
    asyncio.run(run_voice_demo())
