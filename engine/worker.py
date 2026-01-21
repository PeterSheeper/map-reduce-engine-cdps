import os
import zlib
import asyncio
import argparse
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .models import WorkerRegistration, TaskAssignment, TaskResult, ShuffleData

WORKER_ID = None
MASTER_URL = None
DATA_DIR = None
ADVERTISE_HOST = None
PORT = None

shuffle_received = []
shuffle_count = 0
shuffle_expected = 0
shuffle_done = asyncio.Event()


@asynccontextmanager
async def lifespan(app):
    global WORKER_ID
    reg = WorkerRegistration(host=ADVERTISE_HOST, port=PORT)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{MASTER_URL}/register", json=reg.model_dump())
        WORKER_ID = resp.json()["worker_id"]
        print(f"[{WORKER_ID}] Registered with master")
    yield
    print(f"[{WORKER_ID}] Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.post("/task")
async def receive_task(assignment: TaskAssignment):
    asyncio.create_task(execute_task(assignment))
    return {"status": "accepted"}


@app.post("/shuffle")
async def receive_shuffle(data: ShuffleData):
    global shuffle_count
    shuffle_received.extend(data.data)
    shuffle_count += 1
    print(f"[{WORKER_ID}] Got shuffle from {data.from_worker}: {len(data.data)} items ({shuffle_count}/{shuffle_expected})")
    
    if shuffle_count >= shuffle_expected:
        shuffle_done.set()
    return {"status": "ok"}


async def execute_task(assignment: TaskAssignment):
    global shuffle_received, shuffle_count, shuffle_expected, shuffle_done
    
    shuffle_received = []
    shuffle_count = 0
    shuffle_expected = assignment.num_workers - 1
    shuffle_done = asyncio.Event()
    
    print(f"[{WORKER_ID}] Got task {assignment.task_id} (worker {assignment.worker_index + 1}/{assignment.num_workers})")
    
    try:
        namespace = {}
        exec(assignment.task_code, namespace)
        map_func = namespace.get("map_func")
        shuffle_func = namespace.get("shuffle_func", lambda key: zlib.adler32(str(key).encode()))
        
        # === MAP ===
        print(f"[{WORKER_ID}] MAP starting...")
        map_output = map_func(DATA_DIR, WORKER_ID)
        print(f"[{WORKER_ID}] MAP done: {len(map_output)} pairs")
        
        # === SHUFFLE ===
        print(f"[{WORKER_ID}] SHUFFLE starting...")
        
        # Split data by target worker
        buckets = {i: [] for i in range(assignment.num_workers)}
        for key, value in map_output:
            target = shuffle_func(key) % assignment.num_workers
            buckets[target].append((key, value))
        
        # Keep own data
        my_data = buckets[assignment.worker_index]
        print(f"[{WORKER_ID}] Keeping {len(my_data)} items")
        
        # Send to other workers
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, worker in enumerate(assignment.worker_list):
                if i == assignment.worker_index:
                    continue
                
                items = buckets[i]
                shuffle_data = ShuffleData(
                    task_id=assignment.task_id,
                    from_worker=WORKER_ID,
                    data=items
                )
                url = f"http://{worker['host']}:{worker['port']}/shuffle"
                
                for attempt in range(3):
                    try:
                        await client.post(url, json=shuffle_data.model_dump())
                        print(f"[{WORKER_ID}] Sent {len(items)} items to {worker['worker_id']}")
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(f"[{WORKER_ID}] Retry {attempt+1}/3 for {worker['worker_id']}...")
                            await asyncio.sleep(2)
                        else:
                            print(f"[{WORKER_ID}] Failed to send to {worker['worker_id']}: {e}")
        
        # Wait for other workers
        if shuffle_expected > 0:
            print(f"[{WORKER_ID}] Waiting for {shuffle_expected} workers...")
            await asyncio.wait_for(shuffle_done.wait(), timeout=120.0)
        
        # Combine all data
        all_data = my_data + shuffle_received
        print(f"[{WORKER_ID}] SHUFFLE done: {len(all_data)} items")
        
        # === REDUCE ===
        print(f"[{WORKER_ID}] REDUCE starting...")
        
        # Group by key
        grouped = {}
        for key, value in all_data:
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)
        
        print(f"[{WORKER_ID}] Grouped into {len(grouped)} unique keys")
        
        # Apply reduce function to each key
        reduce_func = namespace.get("reduce_func")
        if reduce_func:
            reduce_results = []
            for key, values in grouped.items():
                reduced_value = reduce_func(key, values)
                reduce_results.append((key, reduced_value))
        else:
            reduce_results = [(k, v) for k, v in grouped.items()]
        
        print(f"[{WORKER_ID}] REDUCE done: {len(reduce_results)} results")
        
        # Send final results to master
        result = TaskResult(
            task_id=assignment.task_id,
            worker_id=WORKER_ID,
            phase="reduce_complete",
            results=reduce_results,
            success=True
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(f"{MASTER_URL}/result", json=result.model_dump())
        
        print(f"[{WORKER_ID}] Task complete! Sent {len(reduce_results)} results to master")
        
    except Exception as e:
        print(f"[{WORKER_ID}] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--master-url", default="http://localhost:8000")
    parser.add_argument("--advertise-host", default="localhost")
    args = parser.parse_args()
    
    MASTER_URL = args.master_url
    DATA_DIR = os.environ.get("DATA_DIR", "./data")
    ADVERTISE_HOST = args.advertise_host
    PORT = args.port
    
    uvicorn.run(app, host=args.host, port=args.port)
