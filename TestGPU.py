
import time
from sdk.client import STANClusterClient

def gpu_code():
    import time
    # This code pretends to use torch
    x = "import torch; print('Using CUDA')"
    return "GPU Job Done"

def cpu_code():
    return "CPU Job Done"

def main():
    client = STANClusterClient(token="supersecret")
    
    print("--- GPU Scheduling Test ---")
    
    # 1. Explicit GPU Request
    print("1. Submitting Explicit GPU Job...")
    r1 = client.submit(cpu_code, gpu=True)
    print(f"   Result: {r1.get('status')} - {r1.get('result')}")

    # 2. Implicit GPU Request (Auto-detect)
    print("2. Submitting Implicit GPU Job (torch keyword)...")
    r2 = client.submit(gpu_code) # Should detect 'torch' string in source
    print(f"   Result: {r2.get('status')} - {r2.get('result')}")
    
    # 3. CPU Job
    print("3. Submitting Standard Job...")
    r3 = client.submit(cpu_code, gpu=False)
    print(f"   Result: {r3.get('status')} - {r3.get('result')}")

if __name__ == "__main__":
    main()
