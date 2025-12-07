import unittest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock

# Mock FastAPI machinery before imports
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.security"] = MagicMock()
sys.modules["fastapi.middleware"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()

from triforce.odin.stan.stan_ollama_binding import create_stan_with_ollama
from triforce.odin.stan.remote_ollama_client import RemoteOllamaClient

class TestBindingIntegration(unittest.TestCase):

    def setUp(self):
        # Patch RemoteOllamaClient to avoid real network calls during test
        self.original_generate = RemoteOllamaClient.generate
        RemoteOllamaClient.generate = AsyncMock(return_value="Mocked Remote Response")
        RemoteOllamaClient.embed = AsyncMock(return_value=[0.0]*768)

    def tearDown(self):
        RemoteOllamaClient.generate = self.original_generate

    def test_full_boot(self):
        """Verify create_stan_with_ollama initializes everything."""
        stan = create_stan_with_ollama()
        self.assertIsNotNone(stan)
        self.assertIsNotNone(stan.memory)
        self.assertIsNotNone(stan.ai) # Should be SmartAIProvider
        
    def test_end_to_end_run(self):
        """Verify stan.run executes without crashing."""
        stan = create_stan_with_ollama()
        
        loop = asyncio.new_event_loop()
        try:
            # We explicitly Mock the parser because ParserBrain usually fails if 
            # the Mocked Remote Response isn't valid JSON
            stan.parser_brain.think = AsyncMock(return_value=MagicMock(request_id="123"))
            
            # Mock Scheduler/Planner
            stan.planning_brain.think = AsyncMock(return_value={})
            stan.scheduler.schedule = MagicMock(return_value=MagicMock(assignments=[], unassigned=[]))
            
            # Mock Persona for output
            stan.persona.generate_narration = AsyncMock(return_value=MagicMock(text="Yes master", tone="happy"))

            res = loop.run_until_complete(stan.run("Test Command"))
            
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["message"], "Yes master")
        finally:
            loop.close()

if __name__ == '__main__':
    unittest.main()
