import requests
import time
import json
import uuid

ODIN_URL = "http://localhost:8080"
TOKEN = "supersecret" # Assuming default from docker-compose
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def run_test():
    print("--- Testing MinIO Integration ---")
    
    # 1. Submit Job
    payload = {
        "code": "def main(x): return x * 2",
        "entrypoint": "main",
        "args": [21],
        "job_type": "compute"
    }
    
    logger_header = f"Test-{uuid.uuid4()}"
    print(f"Submitting job...")
    resp = requests.post(f"{ODIN_URL}/submit", json=payload, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Submit Failed: {resp.text}")
        return
        
    job_id = resp.json()['job_id']
    print(f"Job ID: {job_id}")
    
    # 2. Poll for Completion
    print("Waiting for completion...")
    final_resp = None
    for _ in range(10):
        time.sleep(1)
        r = requests.get(f"{ODIN_URL}/jobs/{job_id}", headers=HEADERS)
        data = r.json()
        if data['status'] in ["COMPLETED", "FAILED"]:
            final_resp = data
            break
            
    if not final_resp:
        print("Job timed out")
        return

    print(json.dumps(final_resp, indent=2))
    
    # 3. Verify Result
    if final_resp['status'] == "COMPLETED" and final_resp['result'] == 42:
        print("[PASS] Execution Correct")
    else:
        print("[FAIL] Execution Incorrect")

    # 4. Verify MinIO usage (checking result_path present in response if we added it to response)
    # Note: JobResponse in models.py has result_path
    if final_resp.get('result_path'):
        print(f"[PASS] Result Path Present: {final_resp['result_path']}")
    else:
        print("[FAIL] Result Path Missing")
        
    # 5. Check logs for upload/download messages (optional manual check)

if __name__ == "__main__":
    run_test()
