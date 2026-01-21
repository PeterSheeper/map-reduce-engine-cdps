from pydantic import BaseModel
from typing import List, Any, Optional, Dict


class WorkerInfo(BaseModel):
    worker_id: str
    host: str
    port: int


class WorkerRegistration(BaseModel):
    host: str
    port: int


class TaskAssignment(BaseModel):
    task_id: str
    task_code: str
    worker_index: int
    num_workers: int
    worker_list: List[Dict[str, Any]]


class ShuffleData(BaseModel):
    task_id: str
    from_worker: str
    data: List[Any]  # List of (key, value) pairs


class TaskResult(BaseModel):
    task_id: str
    worker_id: str
    phase: str
    results: List[Any]
    success: bool
    error: Optional[str] = None
