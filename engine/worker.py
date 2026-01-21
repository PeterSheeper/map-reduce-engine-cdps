import os
import asyncio
import argparse
import httpx
from fastapi import FastAPI

from .models import WorkerRegistration, TaskAssignment, TaskResult

WORKER_ID = None
MASTER_URL = None
DATA_DIR = None

app = FastAPI()


@app.on_event("startup")
async def startup():
    global WORKER_ID
    reg = WorkerRegistration(host=ADVERTISE_HOST, port=PORT)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{MASTER_URL}/register", json=reg.model_dump())
        WORKER_ID = resp.json()["worker_id"]
        print(f"[{WORKER_ID}] Registered with master")


@app.post("/task")
async def receive_task(assignment: TaskAssignment):
    asyncio.create_task(execute_task(assignment))
    return {"status": "accepted"}


async def execute_task(assignment: TaskAssignment):
    print(f"[{WORKER_ID}] Executing task {assignment.task_id}")
    
    try:
        namespace = {"data_dir": DATA_DIR, "worker_id": WORKER_ID}
        exec(assignment.task_code, namespace)
        
        run_task = namespace.get("run_task")
        results = run_task(DATA_DIR, WORKER_ID) if run_task else []
        
        result = TaskResult(
            task_id=assignment.task_id,
            worker_id=WORKER_ID,
            results=results,
            success=True
        )
    except Exception as e:
        print(f"[{WORKER_ID}] Error: {e}")
        result = TaskResult(
            task_id=assignment.task_id,
            worker_id=WORKER_ID,
            results=[],
            success=False,
            error=str(e)
        )
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(f"{MASTER_URL}/result", json=result.model_dump())
    
    print(f"[{WORKER_ID}] Done, sent {len(result.results)} results")


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--master-url", default="http://localhost:8000")
    parser.add_argument("--advertise-host", default="localhost")
    args = parser.parse_args()
    
    # Set globals
    MASTER_URL = args.master_url
    DATA_DIR = os.environ.get("DATA_DIR", "./data")
    ADVERTISE_HOST = args.advertise_host
    PORT = args.port
    
    uvicorn.run(app, host=args.host, port=args.port)
