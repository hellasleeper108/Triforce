import requests
import json
import time
import os

ODIN_URL = os.getenv("ODIN_URL", "http://localhost:8080")
API_TOKEN = os.getenv("API_TOKEN", "supersecret")

def run_test():
    print("=== Targeting 'Loki' (MacBook) ===")
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Task: Calculate Primes and Print System Info
    # This identifies if we are on Mac (Darwin) or Linux
    code = """
import platform
import os
import time

def main():
    import platform
    import time
    system = platform.system()
    node = platform.node()
    print(f"Running on {node} ({system})")
    
    # Simulate CPU work
    start = time.time()
    count = 0
    
    # Simple primality test helper
    def is_prime(n):
        if n <= 1: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    for i in range(10000, 20000):
        if is_prime(i):
            count += 1
    duration = time.time() - start
    
    return {
        "node": node,
        "system": system,
        "primes_found": count,
        "duration": duration,
        "message": "Hello from the Metal!" if system == "Darwin" else "Hello from Container!"
    }
"""

    payload = {
        "code": code,
        "entrypoint": "main",
        "args": [],
        "job_type": "compute", # Standard compute favors CPU workers / Low Load
        "requires_gpu": False
    }

    # 1. Submit
    print("[*] Submitting Job...")
    try:
        resp = requests.post(f"{ODIN_URL}/submit", json=payload, headers=headers)
        if resp.status_code != 200:
            print(f"[-] Submission Failed: {resp.text}")
            return
            
        data = resp.json()
        job_id = data["job_id"]
        print(f"    [+] Job ID: {job_id}")
        
    except Exception as e:
        print(f"[-] Connection Failed: {e}")
        return

    # 2. Poll Status
    print("[*] Polling Status...")
    for _ in range(10):
        resp = requests.get(f"{ODIN_URL}/jobs/{job_id}", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            status = data["status"]
            worker = data.get("worker", "unknown")
            
            if status == "COMPLETED":
                print(f"    [+] COMPLETED on {worker}")
                result = data.get("result")
                print(f"    [>] Result: {json.dumps(result, indent=2)}")
                if result and result.get("system") == "Darwin":
                    print("    ✅ SUCCESS: Executed on MacBook (Loki)!")
                else:
                    print("    ⚠️  WARNING: Executed on Linux (Thor/Container).")
                break
            elif status == "FAILED":
                print(f"    [-] FAILED: {data.get('error')}")
                break
            else:
                print(f"    Status: {status}...")
        else:
            print(f"    [-] Poll Failed: {resp.status_code}")
            
        time.sleep(1)
        
if __name__ == "__main__":
    run_test()
