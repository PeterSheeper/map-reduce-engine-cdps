# Map-Reduce Engine

A distributed Map-Reduce engine built with Python and FastAPI.

## Quick Start (Local)

```bash
docker-compose up --build
```

Then open: http://localhost:8000/submit

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
def map_func(data_dir, worker_id):
    """Return list of (key, value) pairs."""
    return [("key1", value1), ("key2", value2), ...]

def shuffle_func(key):
    """Decide which worker handles this key."""
    return zlib.adler32(str(key).encode())

def reduce_func(key, values):
    """Aggregate all values for a key."""
    return sum(values)  # or any aggregation
```
