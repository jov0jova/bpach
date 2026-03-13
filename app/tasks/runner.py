"""
Background task runner using ThreadPoolExecutor.
Tasks write progress to DuckDB tasks table; HTMX polls /tasks/<id>/status.
"""
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import models as m

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="caf_worker")

# ── Cooperative cancellation ──────────────────────────────────────────────────
# Maps task_id → threading.Event. Setting the event signals the worker to stop
# between jobs. Workers that have already started a unit of work finish it first.
_stop_events: dict[str, threading.Event] = {}


def get_stop_event(task_id: str) -> threading.Event | None:
    """Return the stop event for task_id (None if task is not tracked)."""
    return _stop_events.get(task_id)


def request_cancel(db_path: Path, task_id: str) -> None:
    """Signal a running task to stop and mark it cancelled in the DB."""
    event = _stop_events.get(task_id)
    if event:
        event.set()
    m.update_task(db_path, task_id, status="cancelled",
                  message="Cancelled by user.")


def submit_task(db_path: Path, session_id: str, task_type: str, fn, *args, **kwargs) -> str:
    """
    Create a task record and submit fn(*args, **kwargs) to the thread pool.
    The function fn must accept (task_id, db_path, *args, **kwargs) as its first two args.
    Returns the task_id.
    """
    task_id = m.create_task(db_path, session_id, task_type)
    m.update_task(db_path, task_id, status="running", message="Starting…")

    stop_event = threading.Event()
    _stop_events[task_id] = stop_event

    def _wrapper():
        try:
            fn(task_id, db_path, *args, stop_event=stop_event, **kwargs)
            # Only mark done if not already cancelled
            if not stop_event.is_set():
                m.update_task(db_path, task_id, status="done", message="Completed.")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Task %s failed: %s\n%s", task_id, e, tb)
            m.update_task(db_path, task_id, status="error",
                          error=str(e), message=f"Error: {e}")
        finally:
            _stop_events.pop(task_id, None)

    _executor.submit(_wrapper)
    return task_id
