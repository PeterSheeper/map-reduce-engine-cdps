import asyncio
import uuid
from typing import Dict, List

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from .models import WorkerInfo, WorkerRegistration, TaskAssignment, TaskResult


class TaskSubmission(BaseModel):
    task_code: str


workers: Dict[str, WorkerInfo] = {}
results: Dict[str, List[TaskResult]] = {}

app = FastAPI()


@app.on_event("startup")
async def startup():
    print("[Master] Starting...")


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
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, worker in enumerate(workers.values()):
            assignment = TaskAssignment(
                task_id=task_id,
                task_code=submission.task_code,
                worker_index=idx,
                num_workers=len(workers)
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
    print(f"[Master] Result from {res.worker_id}: {res.results}")
    if res.task_id in results:
        results[res.task_id].append(res)
    return {"status": "ok"}


@app.get("/status")
async def status():
    return {
        "workers": len(workers),
        "results": {k: [r.results for r in v] for k, v in results.items()}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
