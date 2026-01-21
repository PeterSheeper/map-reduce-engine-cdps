from pydantic import BaseModel
from typing import List, Any, Optional


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


class TaskResult(BaseModel):
    task_id: str
    worker_id: str
    results: List[Any]
    success: bool
    error: Optional[str] = None
