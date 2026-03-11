"""
Background task runner using ThreadPoolExecutor.
Tasks write progress to DuckDB tasks table; HTMX polls /tasks/<id>/status.
"""
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import models as m

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="caf_worker")


def submit_task(db_path: Path, session_id: str, task_type: str, fn, *args, **kwargs) -> str:
    """
    Create a task record and submit fn(*args, **kwargs) to the thread pool.
    The function fn must accept (task_id, db_path, *args, **kwargs) as its first two args.
    Returns the task_id.
    """
    task_id = m.create_task(db_path, session_id, task_type)
    m.update_task(db_path, task_id, status="running", message="Starting…")

    def _wrapper():
        try:
            fn(task_id, db_path, *args, **kwargs)
            m.update_task(db_path, task_id, status="done", message="Completed.")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Task %s failed: %s\n%s", task_id, e, tb)
            m.update_task(db_path, task_id, status="error",
                          error=str(e), message=f"Error: {e}")

    _executor.submit(_wrapper)
    return task_id
