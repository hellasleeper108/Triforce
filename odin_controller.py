import requests
import time
import datetime
import sys

# Configuration
CONTROLLER_NAME = "Odin"

class OdinController:
    def __init__(self):
        self.workers = {}  # {worker_name: {host, roles, status, last_heartbeat}}
        
    def _log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{CONTROLLER_NAME}] {message}")

    def register_worker(self, name, host, roles):
        """Register a new worker in the in-memory registry."""
        self.workers[name] = {
            "host": host,
            "roles": roles,
            "status": "UNKNOWN",
            "last_heartbeat": None
        }
        self._log(f"Registered worker '{name}' at {host} with roles: {roles}")

    def get_heartbeat(self, worker_name):
        """Check worker health via /heartbeat endpoint."""
        worker = self.workers.get(worker_name)
        if not worker:
            self._log(f"Error: Worker '{worker_name}' not found.")
            return False

        try:
            url = f"{worker['host']}/heartbeat"
            response = requests.get(url, timeout=2)
            
            if response.status_code == 200:
                data = response.json()
                worker["status"] = "ONLINE"
                worker["last_heartbeat"] = time.time()
                cpu = data.get("cpu_load_percent", "N/A")
                mem = data.get("memory_percent", "N/A")
                self._log(f"Heartbeat from '{worker_name}': CPU={cpu}%, Mem={mem}%")
                return True
            else:
                worker["status"] = "ERROR"
                self._log(f"Heartbeat failed for '{worker_name}': {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            worker["status"] = "OFFLINE"
            self._log(f"Worker '{worker_name}' seems offline: {e}")
            return False

    def send_task(self, worker_name, task_name, params=None, requirements=None):
        """Dispatch a task to a worker."""
        if params is None: params = {}
        if requirements is None: requirements = {}
        
        worker = self.workers.get(worker_name)
        if not worker:
            self._log(f"Error: Worker '{worker_name}' not found.")
            return None

        # Basic Check: Is worker online?
        if worker["status"] == "OFFLINE":
             self._log(f"Cannot send task to '{worker_name}': Worker is OFFLINE.")
             return None

        payload = {
            "task_id": f"task-{int(time.time()*1000)}",
            "task": task_name,
            "params": params,
            "requirements": requirements
        }

        try:
            url = f"{worker['host']}/task"
            self._log(f"Sending task '{task_name}' to '{worker_name}'...")
            
            response = requests.post(url, json=payload, timeout=65) # Long timeout for inference
            
            if response.status_code == 200:
                result = response.json()
                self._log(f"Task completed by '{worker_name}'. Result: {result.get('result')}")
                return result
            else:
                try:
                    err = response.json().get("error", "Unknown error")
                except:
                    err = response.text
                self._log(f"Task rejected/failed by '{worker_name}': {err}")
                return None

        except requests.exceptions.RequestException as e:
            self._log(f"Communication error with '{worker_name}': {e}")
            return None

def main():
    odin = OdinController()
    
    # 1. Register Loki
    # Assuming Loki is running on port 8001 from previous step
    odin.register_worker("loki", "http://localhost:8001", ["preprocessing", "inference"])

    # 2. Check Heartbeat
    print("-" * 40)
    odin.get_heartbeat("loki")
    
    # 3. Send Valid Task: Preprocessing
    print("-" * 40)
    odin.send_task("loki", "preprocess_text", {"text": "   HELLO   World  "})
    
    # 4. Send Valid Task: Run Small Inference (Mock or Ollama)
    print("-" * 40)
    odin.send_task("loki", "run_small_model_inference", 
                  {"prompt": "Why is the sky blue?", "model": "llama3.2:1b"},
                  requirements={"min_memory_gb": 2})

    # 5. Send Invalid Task: OOM
    print("-" * 40)
    odin.send_task("loki", "preprocess_text", {"text": "fail"}, requirements={"min_memory_gb": 12})
    
    # 6. Send Invalid Task: GPU Required
    print("-" * 40)
    odin.send_task("loki", "run_small_model_inference", {}, requirements={"gpu": True})

if __name__ == "__main__":
    main()
