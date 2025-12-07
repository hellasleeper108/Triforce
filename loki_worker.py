import http.server
import socketserver
import json
import time
import datetime
import threading
import sys
import psutil
import requests
import os

# Configuration
WORKER_NAME = "loki"
HOST = "0.0.0.0"
PORT = 8001
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Resource Limits
MAX_MEMORY_GB = 10.0
HAS_GPU = False

class LokiHandler(http.server.BaseHTTPRequestHandler):
    def _log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_GET(self):
        if self.path == '/heartbeat':
            self.handle_heartbeat()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == '/task':
            self.handle_task()
        else:
            self.send_error(404, "Not Found")

    def handle_heartbeat(self):
        mem = psutil.virtual_memory()
        cpu_load = psutil.cpu_percent(interval=None)
        
        status = {
            "worker": WORKER_NAME,
            "status": "ACTIVE",
            "cpu_load_percent": cpu_load,
            "memory_percent": mem.percent,
            "memory_available_gb": round(mem.available / (1024**3), 2)
        }
        self._send_json(200, status)

    def handle_task(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            request = json.loads(post_data)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        task_id = request.get("task_id")
        task_name = request.get("task")
        params = request.get("params", {})
        requirements = request.get("requirements", {})

        self._log(f"Received Task {task_id}: {task_name}")

        # 1. Resource Validation
        # 1. Resource Validation
        validation = self.check_resource_limits(params, requirements)
        if not validation["ok"]:
             self._log(f"REJECTED Task {task_id}: {validation['reason']}")
             self._send_json(400, {"status": "REJECTED", "error": validation["reason"]})
             return

    def check_resource_limits(self, params, requirements):
        # 1. Check GPU
        if requirements.get("gpu", False) and not HAS_GPU:
            return {"ok": False, "reason": "GPU required but not available"}
        
        # 2. Check Memory
        req_mem = requirements.get("min_memory_gb", 0) or params.get("memory_gb", 0)
        if req_mem > MAX_MEMORY_GB:
            return {"ok": False, "reason": f"Memory limit exceeded ({req_mem}GB > {MAX_MEMORY_GB}GB)"}

        # 3. Check Model Size (e.g. "llama3:70b")
        model = params.get("model", "")
        if model:
             # Extract size if present (simple heuristic)
             import re
             match = re.search(r'(\d+)b', model.lower())
             if match:
                 size = int(match.group(1))
                 if size > 13:
                     return {"ok": False, "reason": f"Model too large ({size}B > 13B Limit)"}
        
        return {"ok": True}

        # 2. Dispatch
        try:
            result = self.dispatch_task(task_name, params)
            self._log(f"COMPLETED Task {task_id}")
            self._send_json(200, {"status": "COMPLETED", "result": result, "task_id": task_id})
        except Exception as e:
            self._log(f"FAILED Task {task_id}: {str(e)}")
            self._send_json(500, {"status": "FAILED", "error": str(e), "task_id": task_id})

    def dispatch_task(self, task_name, params):
        if task_name == "preprocess_text":
            return self.preprocess_text(params.get("text", ""))
        elif task_name == "generate_small_embeddings":
            return self.generate_small_embeddings(params.get("text", ""))
        elif task_name == "run_small_model_inference":
            return self.run_small_model_inference(
                params.get("prompt", ""), 
                params.get("model", "llama3.2:1b") # Default to small model
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")

    # --- Task Handlers ---

    def preprocess_text(self, text):
        # Example: normalize whitespace and lowercase
        if not text: return ""
        cleaned = " ".join(text.split()).lower()
        return {"processed_text": cleaned, "length": len(cleaned)}

    def generate_small_embeddings(self, text):
        # Mock embedding generation or call Ollama
        # Using simple deterministic mock for "lightweight" demo sans external dep check
        # But user mentioned "Ollama or similar", let's try calling it if env var set, else mock
        
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model": "nomic-embed-text",
                "prompt": text
            }, timeout=5)
            if resp.status_code == 200:
                return {"embedding": resp.json().get("embedding")}
        except:
            pass
            
        # Fallback Mock
        simulated_vector = [ord(c) % 10 / 10.0 for c in text[:10]]
        return {"embedding": simulated_vector, "mock": True}

    def run_small_model_inference(self, prompt, model):
        # Call Ollama
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            
            if resp.status_code == 200:
                return {"response": resp.json().get("response")}
            else:
                raise RuntimeError(f"Ollama Error: {resp.text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Inference Backend Unavailable: {e}")

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle requests in a separate thread."""

def run_server():
    server_address = (HOST, PORT)
    httpd = ThreadedHTTPServer(server_address, LokiHandler)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Loki Worker Online")
    print(f"[{timestamp}] Identity: {WORKER_NAME}")
    print(f"[{timestamp}] Listen: {HOST}:{PORT}")
    print(f"[{timestamp}] Capabilities: GPU={HAS_GPU}, MaxMem={MAX_MEMORY_GB}GB")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    
    httpd.server_close()
    print(f"\n[{timestamp}] Loki Worker Offline")

if __name__ == "__main__":
    run_server()
