import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import random
import time
import json
import os

import psutil

# --- Logging Setup ---
LOG_FILE = "dashboard.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard")

# Import STAN Supernova
from triforce.odin.stan.supernova import STANSupernova

app = FastAPI(title="STAN Dashboard")

# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    # Optional: Rotate or archive log?
    pass

# Setup Templates
templates = Jinja2Templates(directory="triforce/odin/dashboard/templates")

# Global STAN Instance (Mocked for dashboard demo)
stan = STANSupernova(use_mock=False)

class CommandRequest(BaseModel):
    text: str

class ModelSwitchRequest(BaseModel):
    model_name: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/cluster/state")
async def get_cluster_state():
    """
    Returns REAL live telemetry from the local host.
    """
    cpu_usage = psutil.cpu_percent(interval=None)
    mem_info = psutil.virtual_memory()
    
    # Simulate GPU for now if not available, or just set to 0
    gpu_usage = 0 
    
    return {
        "nodes": [
            {
                "id": "odin", 
                "role": "Master", 
                "status": "Online", 
                "cpu": cpu_usage, 
                "ram": mem_info.percent, 
                "gpu": gpu_usage
            }
        ],
        "tasks": [], # No real task tracking yet
        "alerts": []
    }

@app.post("/api/stan/launch")
async def launch_stan():
    """
    Triggers the STAN wake-up sequence.
    """
    print("[Dashboard] Launching STAN...")
    # Simulate a boot sequence output
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        await stan.act("System Startup")
    
    output = f.getvalue()
    return {"message": "STAN Activated", "details": output}

@app.post("/api/commands")
async def send_command(cmd: CommandRequest):
    """
    Executes a NL command via STAN Supernova.
    """
    print(f"[Dashboard] Received command: {cmd.text}")
    
    # Use the returned transcript directly
    output = await stan.act(cmd.text)
    return {"response": output}

@app.get("/api/stan/models")
async def get_models():
    """
    Returns list of available AI models and their active status.
    """
    return {"models": stan.get_models()}

@app.post("/api/stan/models")
async def set_model(req: ModelSwitchRequest):
    """
    Switches the active AI model.
    """
    success = stan.set_model(req.model_name)
    if success:
         return {"message": f"Switched to {req.model_name}", "active_model": req.model_name}
    return JSONResponse(status_code=400, content={"message": "Failed to switch model"})

@app.get("/api/events")
async def event_stream(request: Request):
    """
    SSE Endpoint for real-time logs.
    """
    async def log_generator():
        # Open the log file and tail it
        try:
            with open(LOG_FILE, "r") as f:
                # Seek to end to only show new logs? Or show last N?
                # Let's show last 1000 bytes for context, then tail
                f.seek(0, 2)
                f_size = f.tell()
                f.seek(max(f_size - 2000, 0))
                
                while True:
                    if await request.is_disconnected():
                        break
                        
                    line = f.readline()
                    if line:
                        # Parse standard log line to JSON if possible, or just send raw
                        # Format: 2023-10-27 10:00:00,123 - logger - INFO - message
                        try:
                            parts = line.split(" - ")
                            if len(parts) >= 4:
                                payload = {
                                    "timestamp": parts[0],
                                    "logger": parts[1],
                                    "level": parts[2],
                                    "message": " - ".join(parts[3:]).strip()
                                }
                                yield f"data: {json.dumps({'logs': [payload]})}\n\n"
                        except Exception:
                            pass # Skip malformed lines
                    else:
                        await asyncio.sleep(0.5)
        except FileNotFoundError:
             yield f"data: {json.dumps({'logs': [{'timestamp': '', 'level': 'WARN', 'message': 'Log file not found yet.'}]})}\n\n"
             while True:
                 await asyncio.sleep(1)

    return StreamingResponse(log_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
