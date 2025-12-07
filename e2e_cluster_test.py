import requests
import json
import time
import sys

LOKI_URL = "http://localhost:8001"

def run_test():
    print("=== Loki Cluster E2E Test ===")
    
    # 1. Heartbeat
    try:
        print("[*] Pinging Loki...")
        resp = requests.get(f"{LOKI_URL}/heartbeat", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            print(f"    [+] Loki Online: CPU={data.get('cpu_load_percent')}% Mem={data.get('memory_percent')}%")
        else:
            print(f"    [-] Heartbeat Failed: {resp.status_code}")
            return
    except Exception as e:
        print(f"    [!] Connection Error: {e}")
        print("    Ensure loki_worker.py is running!")
        return

    # 2. Send Task
    task_payload = {
        "task_id": "e2e-test-001",
        "task": "preprocess_text",
        "params": {
            "text": "   Triforce   Cluster   IS   Online   "
        },
        "requirements": {
            "min_memory_gb": 0.5
        }
    }
    
    try:
        print("\n[*] Sending Task: preprocess_text...")
        start = time.time()
        resp = requests.post(f"{LOKI_URL}/task", json=task_payload, timeout=5)
        duration = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"    [+] Task Completed in {duration:.2f}s")
            print(f"    [+] Output: {json.dumps(result.get('result'), indent=2)}")
        else:
            print(f"    [-] Task Rejected/Failed: {resp.status_code}")
            try:
                err_data = resp.json()
                print(f"    [-] Reason: {err_data.get('error', resp.text)}")
            except:
                print(f"    [-] Reason: {resp.text}")
            
    except Exception as e:
        print(f"    [!] Task Error: {e}")

if __name__ == "__main__":
    run_test()
