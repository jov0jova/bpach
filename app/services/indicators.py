"""
Phase 5: Indicator enrichment service.
Applies BaseStrategy.populate_indicators() to every downloaded parquet file.

Parallelism strategy
────────────────────
• Linux / macOS — ProcessPoolExecutor (fork):
    Each job runs in its own OS process, bypassing the GIL entirely.
    Fork copies the parent's already-initialised memory so the Flask app's
    __init__.py is NOT re-executed in worker processes.

• Windows — ThreadPoolExecutor (spawn-safe):
    ProcessPoolExecutor on Windows uses 'spawn', which re-imports every
    module in each worker process.  That triggers app/__init__.py →
    models.init_db() → DuckDB file-lock collision → worker crash.
    Threads share the same process so there is no re-import.  We still get
    meaningful parallelism because pandas / numpy / pyarrow release the GIL
    for most of their heavy operations.

In both cases a try/except catches a broken pool and retries with threads
so the task never fails silently.
"""
import logging
import os
import sys
import threading
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed,
)
from pathlib import Path

import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy
from ..utils.parquet import parquet_path, write_enriched

logger = logging.getLogger(__name__)

# ── Executor selection ────────────────────────────────────────────────────────
# Windows uses 'spawn' for new processes → Flask re-import → DuckDB lock crash.
# Linux/macOS use 'fork' → safe to use processes.
_USE_PROCESSES = sys.platform != "win32"
_WORKERS = min(os.cpu_count() or 1, 16)


# ── Worker function (module-level so ProcessPoolExecutor can pickle it) ───────

def _enrich_worker(job: tuple) -> tuple[str, str, str]:
    """
    Execute in a worker (process or thread): read → indicators → write.
    Returns (symbol, tf, status).  Never touches DuckDB.
    """
    session_id, parquet_dir_str, symbol, tf = job
    path = parquet_path(Path(parquet_dir_str), session_id, symbol, tf)
    if not path.exists():
        return symbol, tf, "skipped"
    try:
        df = pd.read_parquet(path)
        df = BaseStrategy().populate_indicators(df)
        write_enriched(Path(parquet_dir_str), session_id, symbol, tf, df)
        return symbol, tf, "ok"
    except Exception as exc:
        return symbol, tf, f"error: {exc}"


# ── Internal helper ───────────────────────────────────────────────────────────

def _run_jobs(executor_cls, workers: int, jobs: list,
              task_id: str, db_path: Path, total: int) -> int:
    """Submit all jobs to *executor_cls*, collect results, update progress.
    Returns the number of completed jobs."""
    completed = 0
    lock = threading.Lock()          # only needed for ThreadPoolExecutor
    with executor_cls(max_workers=workers) as exe:
        futures = {exe.submit(_enrich_worker, job): job for job in jobs}
        for fut in as_completed(futures):
            symbol, tf, _status = fut.result()
            with lock:
                completed += 1
                m.update_task(db_path, task_id,
                              progress=completed, total=total,
                              message=f"Enriched {symbol} {tf} ({completed}/{total})…")
    return completed


# ── Public entry point ────────────────────────────────────────────────────────

def run_indicators(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list) -> None:
    """Background task: enrich all parquet files with indicators."""

    pairs  = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    total  = len(active) * len(timeframes)

    if total == 0:
        m.update_task(db_path, task_id, progress=0, total=0,
                      message="No pairs to enrich.")
        return

    jobs    = [(session_id, str(parquet_dir), p["symbol"], tf)
               for p in active for tf in timeframes]
    workers = min(_WORKERS, total)

    if _USE_PROCESSES:
        # Attempt true multi-process parallelism; fall back to threads if the
        # pool crashes (e.g. OOM on low-memory machines).
        try:
            _run_jobs(ProcessPoolExecutor, workers, jobs,
                      task_id, db_path, total)
        except Exception as exc:
            logger.warning(
                "ProcessPoolExecutor failed (%s) — retrying with threads.", exc)
            m.update_task(db_path, task_id, progress=0, total=total,
                          message="Process pool failed, retrying with threads…")
            _run_jobs(ThreadPoolExecutor, workers, jobs,
                      task_id, db_path, total)
    else:
        # Windows: threads only — avoids spawn re-import / DuckDB lock crash.
        _run_jobs(ThreadPoolExecutor, workers, jobs,
                  task_id, db_path, total)

    m.update_task(db_path, task_id, progress=total, total=total,
                  message=(f"Done — indicators added to "
                           f"{len(active)} pairs × {len(timeframes)} timeframes."))
