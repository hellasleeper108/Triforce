import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import random
import time

# Import STAN Supernova
# In a real deployment, this would attach to the existing running instance.
# Here we instantiate a fresh one (mocked) for the API to control.
from triforce.odin.stan.supernova import STANSupernova

app = FastAPI(title="STAN Dashboard")

# Setup Templates
templates = Jinja2Templates(directory="triforce/odin/dashboard/templates")

# Global STAN Instance (Mocked for dashboard demo)
stan = STANSupernova(use_mock=True)

class CommandRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/cluster/state")
async def get_cluster_state():
    """
    Returns simulated live telemetry for the frontend.
    """
    # Simulate dynamic changes
    is_stress = random.random() > 0.8
    thor_cpu = random.randint(80, 99) if is_stress else random.randint(20, 40)
    
    return {
        "nodes": [
            {"id": "odin", "role": "Master", "status": "Online", "cpu": 15, "ram": 45, "gpu": 0},
            {"id": "thor", "role": "Worker", "status": "Online", "cpu": thor_cpu, "ram": 70, "gpu": 95 if is_stress else 10},
            {"id": "loki", "role": "Worker", "status": "Online", "cpu": 10, "ram": 25, "gpu": 5},
        ],
        "tasks": [
            {"id": "task-a1", "node": "thor", "status": "Running", "elapsed": "12m 30s"},
            {"id": "task-b2", "node": "odin", "status": "Running", "elapsed": "45s"},
            {"id": "task-c3", "node": "loki", "status": "Pending", "elapsed": "0s"},
        ],
        "alerts": [
            {"level": "warning", "msg": "Thor GPU Temp > 85C"} if is_stress else None,
            {"level": "info", "msg": "Autoscaler active"}
        ]
    }

@app.post("/api/commands")
async def send_command(cmd: CommandRequest):
    """
    Executes a NL command via STAN Supernova.
    """
    print(f"[Dashboard] Received command: {cmd.text}")
    
    # We capture logs to return as "response"
    # This is a hacky way to pipe stdout back to UI for the demo
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        await stan.act(cmd.text)
    
    output = f.getvalue()
    return {"response": output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
