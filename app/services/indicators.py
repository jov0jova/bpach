"""
Phase 5: Indicator enrichment service.
Applies BaseStrategy.populate_indicators() to every downloaded parquet file.
"""
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy
from ..utils.parquet import parquet_path, write_enriched

logger = logging.getLogger(__name__)

# Number of parallel workers — capped at 8 to avoid excessive memory use
_WORKERS = min(8, os.cpu_count() or 4)


def run_indicators(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list) -> None:
    """Background task: enrich all parquet files with indicators (parallel)."""

    pairs = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]

    total = len(active) * len(timeframes)
    if total == 0:
        m.update_task(db_path, task_id, progress=0, total=0,
                      message="No pairs to enrich.")
        return

    completed = 0
    lock = threading.Lock()

    def enrich_one(pair: dict, tf: str) -> tuple[str, str, str]:
        symbol = pair["symbol"]
        path = parquet_path(parquet_dir, session_id, symbol, tf)
        if not path.exists():
            return symbol, tf, "skipped"
        try:
            strat = BaseStrategy()
            df = pd.read_parquet(path)
            df = strat.populate_indicators(df)
            write_enriched(parquet_dir, session_id, symbol, tf, df)
            logger.debug("Enriched %s %s (%d rows)", symbol, tf, len(df))
            return symbol, tf, "ok"
        except Exception as e:
            logger.warning("Failed to enrich %s %s: %s", symbol, tf, e)
            return symbol, tf, f"error: {e}"

    workers = min(_WORKERS, total)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(enrich_one, pair, tf): (pair["symbol"], tf)
            for pair in active
            for tf in timeframes
        }
        for fut in as_completed(futures):
            symbol, tf, _status = fut.result()
            with lock:
                completed += 1
                m.update_task(db_path, task_id, progress=completed, total=total,
                              message=f"Enriched {symbol} {tf} ({completed}/{total})…")

    m.update_task(db_path, task_id, progress=total, total=total,
                  message=f"Indicators added to {len(active)} pairs × {len(timeframes)} timeframes.")
