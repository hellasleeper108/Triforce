import asyncio
from triforce.odin.stan.ai_provider import AIProviderFactory, ProviderConfig

async def test_integration():
    print("--- Testing AIProvider Integration ---")
    config = ProviderConfig(
        type="ollama",
        base_url="http://mock-url:11434"
    )
    
    provider = AIProviderFactory.create(config)
    print(f"Factory created: {type(provider)}")
    print(f"Base URL: {provider.base_url}")
    
    # Verify module location
    print(f"Module: {provider.__module__}")

    try:
        from triforce.odin.stan.ollama_provider import OllamaProvider
        assert isinstance(provider, OllamaProvider)
        print("SUCCESS: Instance is correctly of type triforce.odin.stan.ollama_provider.OllamaProvider")
    except AssertionError:
        print("FAILURE: Type mismatch")
    except ImportError:
        print("FAILURE: Could not import new module")

if __name__ == "__main__":
    asyncio.run(test_integration())
