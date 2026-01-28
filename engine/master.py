import uuid
from typing import Dict, List
from contextlib import asynccontextmanager

import httpx
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from .models import WorkerInfo, WorkerRegistration, TaskAssignment, TaskResult

SUBMIT_TIMEOUT = 300.0


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
    
    async with httpx.AsyncClient(timeout=SUBMIT_TIMEOUT) as client:
        async def send_to_worker(idx, worker):
            assignment = TaskAssignment(
                task_id=task_id,
                task_code=submission.task_code,
                worker_index=idx,
                num_workers=len(workers),
                worker_list=worker_list
            )
            try:
                print(f"[Master] Task sent to {worker.worker_id}")
                await client.post(f"http://{worker.host}:{worker.port}/task", json=assignment.model_dump())
            except Exception as e:
                print(f"[Master] Failed: {e}")
        
        await asyncio.gather(*[send_to_worker(idx, w) for idx, w in enumerate(workers.values())])
    
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
        "results": {k: [{
            "worker": r.worker_id, 
            "phase": r.phase, 
            "time": r.time, 
            "count": len(r.results),
            "metrics": r.metrics
        } for r in v] for k, v in results.items()}
    }


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in results:
        return {"error": "task not found"}
    
    all_results = []
    for worker_result in results[task_id]:
        if worker_result.phase == "reduce_complete":
            all_results.extend(worker_result.results)

    # Uncomment below for accident analysis to sort by danger score
    # all_results.sort(key=lambda x: x[1].get('danger_score', 0) if isinstance(x[1], dict) else x[1], reverse=True)
    
    return {
        "task_id": task_id,
        "total_items": len(all_results),
        "data": all_results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

