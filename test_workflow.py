import requests
import json
import time
import os

ODIN_URL = os.getenv("ODIN_URL", "http://localhost:8080")
API_TOKEN = os.getenv("API_TOKEN", "supersecret")

def run_test():
    print("=== Testing Agent Workflow Engine ===")
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Define Workflow: Step A -> Step B
    workflow_payload = {
        "steps": [
            {
                "step_id": "step_a",
                "submission": {
                    "job_type": "python",
                    "code": "print('Hello from Step A'); result='Artifact A'",
                    "entrypoint": "python",
                    "args": []
                },
                "dependencies": []
            },
            {
                "step_id": "step_b",
                "submission": {
                    "job_type": "python",
                    "code": "import os; import json; print('Step B'); env=os.environ.get('TRIFORCE_UPSTREAM_ARTIFACTS'); print('Env:', env); artifacts=json.loads(env); assert 'step_a' in artifacts, 'Missing step_a artifact'",
                    "entrypoint": "python",
                    "args": []
                },
                "dependencies": ["step_a"]
            }
        ]
    }
    
    # 1. Submit
    print("[*] Submitting Workflow...")
    resp = requests.post(f"{ODIN_URL}/workflows", json=workflow_payload, headers=headers)
    if resp.status_code != 200:
        print(f"[-] Submission Failed: {resp.text}")
        return
        
    wf_id = resp.json()["workflow_id"]
    print(f"    [+] Workflow ID: {wf_id}")
    
    # 2. Poll Status
    print("[*] Polling Status...")
    for _ in range(20):
        resp = requests.get(f"{ODIN_URL}/workflows/{wf_id}", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            status = data["status"]
            step_statuses = data["step_statuses"]
            print(f"    Status: {status} | Steps: {step_statuses}")
            
            if status in ["COMPLETED", "FAILED"]:
                break
        else:
            print(f"    [-] Poll Failed: {resp.status_code}")
            
        time.sleep(1)
        
    print("=== Test Complete ===")
    
if __name__ == "__main__":
    run_test()
