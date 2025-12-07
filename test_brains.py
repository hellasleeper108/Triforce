import asyncio
import logging
from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig
from triforce.odin.stan.brains import ParsingBrain, PlanningBrain, ReflectionBrain
from triforce.odin.stan.persona import PersonaBrain
from triforce.odin.stan.model_switcher import ModelSwitcher

# Mock Logger
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# Fix for PersonaBrain's required ModelSwitcher
class MockSwitcher:
    def __init__(self): pass
    async def select_model(self, t, c): return "mock-model"

async def test_observability():
    print("--- Brain Observability Test ---")
    
    # Setup
    ai = AIProviderFactory.create(ProviderConfig(type="mock"))
    
    # 1. Parsing Brain
    print("\n[Parsing Brain]")
    parser = ParsingBrain(ai)
    try:
        # Mock provider needs to return valid JSON for the brains to work
        # Since standard mock provider returns strings, this might fail unless we hack it or just verify logs
        # But wait! 'brains.py' expects json_format=True.
        # Our MockProvider in ai_provider.py handles json_format=True by returning '{"status": "mocked"}'
        # This will fail Pydantic validation for TaskGraph (TaskNode expects task_id, etc)
        # So we expect a validation error, but we should see the LATENCY log first.
        await parser.think("Deploy to Thor")
    except Exception as e:
        print(f"caught expected error (mock data mismatch): {e}")

    # 2. Persona Brain
    print("\n[Persona Brain]")
    # Needs valid switcher
    persona = PersonaBrain(ai, MockSwitcher())
    resp = await persona._speak("Hello World")
    print(f"Persona: {resp.text} ({resp.tone})")

if __name__ == "__main__":
    asyncio.run(test_observability())
