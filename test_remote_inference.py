import asyncio
import logging
from triforce.odin.stan.model_registry import ModelRegistry
from triforce.odin.stan.model_routing import ModelRouter

# Configure logging to see retry attempts
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

async def test_remote_integration():
    print("--- Remote Inference Integration Test ---")
    
    registry = ModelRegistry()
    router = ModelRouter(registry)
    
    # 1. Route a request to Thor (Remote)
    print("\n[1] Routing 'llama3:70b' to Thor...")
    route = router.route_generate("Analysis", "llama3:70b")
    
    if route.success and route.selected_node:
        print(f" -> Selected Node: {route.selected_node.hostname}")
        print(f" -> API URL: {route.api_url}")
        
        # 2. Get Client
        client = router.get_client(route.selected_node)
        print(f" -> Client Created: {type(client)}")
        
        # 3. Attempt Call (Expect Retry -> Fail in 0.0.0.0 environment)
        # We lower retries via private member hack for speed of test
        import aiohttp
        client.retries = 1 
        client.timeout = aiohttp.ClientTimeout(total=1)
        
        print(" -> Attempting remote generation (expecting failure)...")
        try:
            await client.generate("Hello Thor")
        except Exception as e:
            print(f" -> Caught Expected Error: {type(e).__name__} - {e}")
            
    await router.close_all()

if __name__ == "__main__":
    asyncio.run(test_remote_integration())
