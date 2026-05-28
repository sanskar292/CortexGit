from fastapi import FastAPI
import asyncio
from api.routes.events import router as events_router
from cortexgit.graph.expiration import start_background_expiration

app = FastAPI(
    title="CortexGit API Server",
    description="FastAPI interface for the CortexGit persistent agent memory SDK",
    version="0.1.0"
)

# Register routes
app.include_router(events_router)

@app.on_event("startup")
async def startup_event():
    # Start the background node expiration cleanup task
    asyncio.create_task(start_background_expiration())

@app.get("/")
async def root():
    return {"message": "Welcome to CortexGit API Server"}
