
import asyncio
import logging
from triforce.odin.stan.ai_provider import ProviderConfig
from triforce.odin.stan.ollama_provider import OllamaProvider

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_ollama_json():
    # Setup provider
    config = ProviderConfig(
        type="ollama",
        base_url="http://localhost:11434",
    )
    provider = OllamaProvider(config)
    
    # Fetch models
    models = provider.get_available_models()
    print(f"Discovered models: {[m['name'] for m in models]}")
    
    for m in models:
        model_name = m["name"]
        print(f"\n=== Testing Model: {model_name} ===")
        
        # Test 1: Simple Generation
        try:
            res = await provider.generate("Why is the sky blue?", model=model_name)
            print(f"Simple Gen Result ({len(res)} chars): {res[:50]}...")
        except Exception as e:
            print(f"Simple Gen Failed: {e}")

        # Test 2: JSON Generation
        system_prompt = "Output JSON: {'status': 'ok'}"
        try:
            res = await provider.generate(
                "Status check.", 
                system=system_prompt, 
                model=model_name,
                json_format=True
            )
            print(f"JSON Gen Result ({len(res)} chars): '{res}'")
        except Exception as e:
            print(f"JSON Gen Failed: {e}") 


if __name__ == "__main__":
    asyncio.run(test_ollama_json())
