"""
Phase 5: Indicator enrichment service.
Applies BaseStrategy.populate_indicators() to every downloaded parquet file.
"""
import logging
from pathlib import Path

import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy
from ..utils.parquet import parquet_path, write_enriched

logger = logging.getLogger(__name__)


def run_indicators(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list) -> None:
    """Background task: enrich all parquet files with indicators."""

    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    pairs = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]

    total = len(active) * len(timeframes)
    step = 0
    strategy = BaseStrategy()

    for pair in active:
        symbol = pair["symbol"]
        for tf in timeframes:
            step += 1
            progress(step, total, f"Enriching {symbol} {tf}…")
            path = parquet_path(parquet_dir, session_id, symbol, tf)
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                df = strategy.populate_indicators(df)
                write_enriched(parquet_dir, session_id, symbol, tf, df)
                logger.debug("Enriched %s %s (%d rows)", symbol, tf, len(df))
            except Exception as e:
                logger.warning("Failed to enrich %s %s: %s", symbol, tf, e)

    progress(total, total, f"Indicators added to {len(active)} pairs × {len(timeframes)} timeframes.")
