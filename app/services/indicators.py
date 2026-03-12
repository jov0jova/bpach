"""
Phase 5: Indicator enrichment service.
Applies BaseStrategy.populate_indicators() to every downloaded parquet file.

Uses ProcessPoolExecutor so each worker runs in its own OS process, bypassing
Python's GIL and saturating all available CPU cores.  The worker function must
be defined at module level (not nested) so it is picklable.
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy
from ..utils.parquet import parquet_path, write_enriched

logger = logging.getLogger(__name__)

# One worker per logical CPU, capped at 16 to bound memory consumption.
# On a 4-core machine this gives 4× the throughput of the old thread pool.
_WORKERS = min(os.cpu_count() or 1, 16)


# ── Worker (module-level so multiprocessing can pickle it) ────────────────────

def _enrich_worker(job: tuple) -> tuple[str, str, str]:
    """
    Execute in a worker process: read parquet → add indicators → write parquet.
    Returns (symbol, tf, status).  Never touches DuckDB — caller handles that.
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


# ── Public entry point ────────────────────────────────────────────────────────

def run_indicators(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list) -> None:
    """Background task: enrich all parquet files with indicators (multi-process)."""

    pairs  = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    total  = len(active) * len(timeframes)

    if total == 0:
        m.update_task(db_path, task_id, progress=0, total=0,
                      message="No pairs to enrich.")
        return

    # Build job list — plain tuples, fully picklable
    jobs = [
        (session_id, str(parquet_dir), pair["symbol"], tf)
        for pair in active
        for tf in timeframes
    ]

    workers   = min(_WORKERS, total)
    completed = 0

    # ProcessPoolExecutor: each future runs in a separate OS process.
    # Progress is updated in the main process after each result arrives.
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_enrich_worker, job): job for job in jobs}
        for fut in as_completed(futures):
            symbol, tf, _status = fut.result()
            completed += 1
            m.update_task(db_path, task_id,
                          progress=completed, total=total,
                          message=f"Enriched {symbol} {tf} ({completed}/{total})…")

    m.update_task(db_path, task_id, progress=total, total=total,
                  message=(f"Done — indicators added to "
                           f"{len(active)} pairs × {len(timeframes)} timeframes."))
