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

# Base OHLCV columns present in every parquet before indicator enrichment.
_OHLCV_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

logger = logging.getLogger(__name__)

# ── Executor selection ────────────────────────────────────────────────────────
# Windows uses 'spawn' for new processes → Flask re-import → DuckDB lock crash.
# Linux/macOS use 'fork' → safe to use processes.
_USE_PROCESSES = sys.platform != "win32"
# No artificial cap — use all logical CPUs. On Windows (threads) pandas/numpy
# release the GIL for heavy ops so more threads = more throughput.
_WORKERS = os.cpu_count() or 1


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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _reset_worker(job: tuple) -> tuple[str, str, str]:
    """Strip all non-OHLCV columns from a parquet file in-place."""
    session_id, parquet_dir_str, symbol, tf = job
    path = parquet_path(Path(parquet_dir_str), session_id, symbol, tf)
    if not path.exists():
        return symbol, tf, "skipped"
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        keep   = [c for c in schema.names if c in _OHLCV_COLS]
        if len(keep) == len(schema.names):
            return symbol, tf, "already_clean"
        df = pd.read_parquet(path, columns=keep)
        df.reset_index(drop=True).to_parquet(path, index=False)
        return symbol, tf, "ok"
    except Exception as exc:
        return symbol, tf, f"error: {exc}"


def _run_jobs(executor_cls, workers: int, jobs: list,
              task_id: str, db_path: Path, total: int,
              worker_fn=None, stop_event: threading.Event | None = None) -> int:
    """Submit all jobs to *executor_cls*, collect results, update progress.
    Checks stop_event between completions — sets pending futures cancelled and
    returns early when signalled.  Returns the number of completed jobs."""
    if worker_fn is None:
        worker_fn = _enrich_worker
    completed = 0
    lock = threading.Lock()
    with executor_cls(max_workers=workers) as exe:
        futures = [exe.submit(worker_fn, job) for job in jobs]
        fut_map = {f: jobs[i] for i, f in enumerate(futures)}
        for fut in as_completed(fut_map):
            if stop_event and stop_event.is_set():
                # Cancel every future that hasn't started yet
                for f in futures:
                    f.cancel()
                break
            try:
                symbol, tf, _status = fut.result()
            except Exception:
                symbol, tf = "?", "?"
            with lock:
                completed += 1
                m.update_task(db_path, task_id,
                              progress=completed, total=total,
                              message=f"Processed {symbol} {tf} ({completed}/{total})…")
    return completed


# ── Public entry point ────────────────────────────────────────────────────────

def _dispatch(executor_cls_primary, workers, jobs, task_id, db_path, total,
              worker_fn=None, stop_event=None):
    """Run jobs with the primary executor, fall back to threads on failure."""
    kwargs = dict(worker_fn=worker_fn, stop_event=stop_event)
    if _USE_PROCESSES and executor_cls_primary is ProcessPoolExecutor:
        try:
            return _run_jobs(ProcessPoolExecutor, workers, jobs,
                             task_id, db_path, total, **kwargs)
        except Exception as exc:
            logger.warning("ProcessPoolExecutor failed (%s) — retrying with threads.", exc)
            m.update_task(db_path, task_id, progress=0, total=total,
                          message="Process pool failed, retrying with threads…")
    return _run_jobs(ThreadPoolExecutor, workers, jobs,
                     task_id, db_path, total, **kwargs)


def run_indicators(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list,
                   stop_event: threading.Event | None = None) -> None:
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

    _dispatch(ProcessPoolExecutor, workers, jobs, task_id, db_path, total,
              worker_fn=_enrich_worker, stop_event=stop_event)

    if stop_event and stop_event.is_set():
        return   # status already set to 'cancelled' by runner.request_cancel

    m.update_task(db_path, task_id, progress=total, total=total,
                  message=(f"Done — indicators added to "
                           f"{len(active)} pairs × {len(timeframes)} timeframes."))


def run_reset_indicators(task_id: str, db_path: Path, session_id: str,
                         parquet_dir: Path, timeframes: list,
                         stop_event: threading.Event | None = None) -> None:
    """Background task: strip indicator columns from all parquet files,
    keeping only the original OHLCV columns."""
    pairs  = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    total  = len(active) * len(timeframes)

    if total == 0:
        m.update_task(db_path, task_id, progress=0, total=0,
                      message="No pairs to reset.")
        return

    jobs    = [(session_id, str(parquet_dir), p["symbol"], tf)
               for p in active for tf in timeframes]
    workers = min(_WORKERS, total)

    # Reset worker uses pandas only (no BaseStrategy) — threads are fine on all platforms.
    _run_jobs(ThreadPoolExecutor, workers, jobs, task_id, db_path, total,
              worker_fn=_reset_worker, stop_event=stop_event)

    if stop_event and stop_event.is_set():
        return

    m.update_task(db_path, task_id, progress=total, total=total,
                  message=(f"Indicators removed from "
                           f"{len(active)} pairs × {len(timeframes)} timeframes."))
