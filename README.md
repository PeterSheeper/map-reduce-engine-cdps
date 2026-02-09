# Map-Reduce Engine

A distributed Map-Reduce engine built with Python and FastAPI.

## Quick Start (Local - Docker)

```bash
docker-compose up --build
```

## Manual Start

### MASTER
```bash
python -m engine.master
```
### WORKER
```bash
python -m engine.worker --master-url http://MASTER_IP:8000 --advertise-host THIS_IP --port PORT --data-dir DATA_DIR
```
MASTER_IP - IP address of the machine running the master

THIS_IP - IP address of the machine running the worker

PORT - port of the worker (on one machine has to be different for each worker and distict from master port e.g 8001 or 8002)

DATA_DIR - path to the directory with data for specific worker (has to be different for each worker)

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/submit` | GET | Run example word count task |
| `/submit` | POST | Submit custom task (JSON: `{"task_code": "..."}`) |
| `/status` | GET | View workers and task status |
| `/results/{task_id}` | GET | Get final results |

## Writing Tasks

Tasks must define these functions in a Python file:

```python
def init_func(worker_id, master_url, data_dir, advertise_host, port):
    """(optional) Makes init operations based on worker data."""
    # make any init operations you want, having worker configuration
    # this method is optional, you don't need to implement it

def map_func(data_dir, worker_id):
    """Return list of (key, value) pairs."""
    return [("key1", value1), ("key2", value2), ...]

def shuffle_func(key):
    """Decide which workers handle this key."""
    return [zlib.adler32(str(key).encode()), ...]

def reduce_func(data, worker_id):
    """Processes data (in format: List[dict_item[key, values]]."""
    results = []
    for key, values in data:
        results.append((key, sum(values))) # or any other aggregation
    return results
```
