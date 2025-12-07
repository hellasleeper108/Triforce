import unittest
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from triforce.odin.stan.ai_provider import AIProvider
from triforce.odin.stan.brains import ParsingBrain, PlanningBrain

class MockAI(AIProvider):
    def __init__(self, config=None):
        self.config = config or MagicMock()
        self.logger = logging.getLogger("mock.ai")

    async def generate(self, prompt, **kwargs):
        system = kwargs.get("system", "")
        if "Command:" in prompt or "PARSING" in system:
            return '{"tasks": [{"task_id": "1", "high_level_action": "TEST"}]}'
        if "Optimize" in prompt or "PLANNING" in system:
            return '{"plan_id": "p1", "schedule": []}'
        return "Generic Response"

    async def embed(self, text, model=None):
        return [0.1, 0.2, 0.3]
    
    async def classify(self, text, labels, model=None):
        return labels[0]

class TestBrains(unittest.TestCase):
    
    def setUp(self):
        self.ai = MockAI()
        
    def test_parsing_brain(self):
        brain = ParsingBrain(self.ai)
        loop = asyncio.new_event_loop()
        graph = loop.run_until_complete(brain.think("Do a test"))
        loop.close()
        
        self.assertEqual(len(graph.tasks), 1)
        self.assertEqual(graph.tasks[0].high_level_action, "TEST")

    def test_planning_brain(self):
        brain = PlanningBrain(self.ai)
        # Mocking input graph object 
        mock_graph = MagicMock()
        mock_graph.dict.return_value = {}
        
        loop = asyncio.new_event_loop()
        plan = loop.run_until_complete(brain.think(mock_graph))
        loop.close()
        
        self.assertEqual(plan["plan_id"], "p1")

if __name__ == '__main__':
    unittest.main()
