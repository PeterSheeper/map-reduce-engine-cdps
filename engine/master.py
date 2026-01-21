import uuid
from typing import Dict, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from .models import WorkerInfo, WorkerRegistration, TaskAssignment, TaskResult


class TaskSubmission(BaseModel):
    task_code: str


workers: Dict[str, WorkerInfo] = {}
results: Dict[str, List[TaskResult]] = {}


@asynccontextmanager
async def lifespan(app):
    print("[Master] Starting...")
    yield
    print("[Master] Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.post("/register")
async def register(reg: WorkerRegistration):
    worker_id = f"worker-{len(workers) + 1}"
    workers[worker_id] = WorkerInfo(worker_id=worker_id, host=reg.host, port=reg.port)
    print(f"[Master] Worker registered: {worker_id} at {reg.host}:{reg.port}")
    return {"worker_id": worker_id}


@app.post("/submit")
async def submit(submission: TaskSubmission):
    task_id = str(uuid.uuid4())[:8]
    
    if not workers:
        print("[Master] No workers!")
        return {"error": "no workers"}
    
    results[task_id] = []
    print(f"[Master] Submitting task {task_id} to {len(workers)} workers")
    
    worker_list = [
        {"worker_id": w.worker_id, "host": w.host, "port": w.port}
        for w in workers.values()
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, worker in enumerate(workers.values()):
            assignment = TaskAssignment(
                task_id=task_id,
                task_code=submission.task_code,
                worker_index=idx,
                num_workers=len(workers),
                worker_list=worker_list
            )
            try:
                await client.post(f"http://{worker.host}:{worker.port}/task", json=assignment.model_dump())
                print(f"[Master] Task sent to {worker.worker_id}")
            except Exception as e:
                print(f"[Master] Failed: {e}")
    
    return {"task_id": task_id}


@app.get("/submit")
async def submit_test():
    with open("tasks/example_wordcount.py", "r") as f:
        task_code = f.read()
    return await submit(TaskSubmission(task_code=task_code))


@app.post("/result")
async def result(res: TaskResult):
    print(f"[Master] {res.phase} from {res.worker_id}: {len(res.results)} items")
    if res.task_id in results:
        results[res.task_id].append(res)
    return {"status": "ok"}


@app.get("/status")
async def status():
    return {
        "workers": len(workers),
        "worker_list": [{"id": w.worker_id, "host": w.host, "port": w.port} for w in workers.values()],
        "results": {k: [{"worker": r.worker_id, "phase": r.phase, "count": len(r.results)} for r in v] for k, v in results.items()}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
