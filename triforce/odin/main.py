import uvicorn
import asyncio
import os
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from triforce.odin.controllers.worker import ClusterManager
from triforce.odin.scheduler.core import Scheduler
from triforce.odin.api import routes
from triforce.odin.utils.logger import logger

PORT = int(os.getenv("PORT", 8080))
API_TOKEN = os.getenv("API_TOKEN", "default-insecure-token")
security = HTTPBearer()

async def verify_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    
    role = request.headers.get("X-Role", "unknown")
    logger.info(f"Authenticated request from role: {role}", extra={"event": "auth", "role": role})

app = FastAPI(title="ODIN Controller (Triforce)", dependencies=[Depends(verify_token)])

# Initialize components
cluster = ClusterManager()
scheduler = Scheduler(cluster)

# Inject dependencies into router (Naive injection, or better using Depends)
routes.cluster = cluster
routes.scheduler = scheduler

app.include_router(routes.router)

@app.on_event("startup")
async def startup():
    logger.info("ODIN Controller Online", extra={"event": "startup", "role": "controller"})
    asyncio.create_task(cluster.update_health())
    asyncio.create_task(scheduler.run())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None)
