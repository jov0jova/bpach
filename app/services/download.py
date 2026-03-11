"""
Phase 2: Data download service.
Downloads OHLCV data from exchange via CCXT and stores as Parquet files.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

from .. import models as m
from ..utils.parquet import write_ohlcv

logger = logging.getLogger(__name__)

TIMEFRAME_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000,
    "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}


def _get_exchange(exchange_id: str):
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")
    ex = exchange_class({"enableRateLimit": True})
    return ex


def fetch_active_pairs(exchange_id: str, quote_asset: str, min_volume: float = 0) -> list:
    """Fetch all active trading pairs for the given quote asset."""
    ex = _get_exchange(exchange_id)
    markets = ex.load_markets()
    pairs = []
    for symbol, market in markets.items():
        if not market.get("active", True):
            continue
        if market.get("quote") != quote_asset:
            continue
        if market.get("type", "spot") not in ("spot", "future"):
            continue
        pairs.append({
            "symbol": symbol,
            "base_asset": market.get("base", ""),
            "quote_asset": quote_asset,
            "active": True,
            "volume_24h": 0.0,
        })
    return pairs


def run_download(task_id: str, db_path: Path, session_id: str,
                 parquet_dir: Path, exchange_id: str, timeframes: list,
                 quote_asset: str, max_pairs: int = 200,
                 lookback_days: int = 365) -> None:
    """
    Background task: download OHLCV data for all pairs in the session.
    Progress is written to the tasks table.
    """
    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    progress(0, 1, "Connecting to exchange…")

    try:
        ex = _get_exchange(exchange_id)
        ex.load_markets()
    except Exception as e:
        m.update_task(db_path, task_id, status="error", error=str(e),
                      message=f"Failed to connect to {exchange_id}: {e}")
        return

    # Get pairs from DB (already populated during session creation)
    pairs_db = m.list_pairs(db_path, session_id)
    active_pairs = [p for p in pairs_db if p["active"] and not p["excluded"]][:max_pairs]

    if not active_pairs:
        # Fetch from exchange if not yet populated
        try:
            raw_pairs = fetch_active_pairs(exchange_id, quote_asset)[:max_pairs]
            m.upsert_pairs(db_path, session_id, raw_pairs)
            active_pairs = m.list_pairs(db_path, session_id)[:max_pairs]
        except Exception as e:
            m.update_task(db_path, task_id, status="error", error=str(e))
            return

    total_steps = len(active_pairs) * len(timeframes)
    step = 0
    since_ms = int((time.time() - lookback_days * 86400) * 1000)

    for pair in active_pairs:
        symbol = pair["symbol"]
        for tf in timeframes:
            step += 1
            progress(step, total_steps, f"Downloading {symbol} {tf}…")
            try:
                _download_pair(ex, db_path, parquet_dir, session_id,
                               symbol, tf, since_ms)
            except Exception as e:
                logger.warning("Failed to download %s %s: %s", symbol, tf, e)
                continue

    progress(total_steps, total_steps, f"Downloaded {len(active_pairs)} pairs × {len(timeframes)} timeframes.")
    m.update_session_status(db_path, session_id, "data_ready")


def _download_pair(ex, db_path: Path, parquet_dir: Path, session_id: str,
                   symbol: str, timeframe: str, since_ms: int) -> None:
    """Download all historical OHLCV candles for one symbol/timeframe."""
    tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
    limit = 1000
    all_candles = []
    current_since = since_ms

    while True:
        try:
            candles = ex.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
        except Exception as e:
            logger.debug("OHLCV fetch error %s %s: %s", symbol, timeframe, e)
            break

        if not candles:
            break

        all_candles.extend(candles)

        if len(candles) < limit:
            break

        current_since = candles[-1][0] + tf_ms
        # Small delay to respect rate limits
        time.sleep(0.05)

    if not all_candles:
        return

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    write_ohlcv(parquet_dir, session_id, symbol, timeframe, df)

    m.update_pair_data_info(
        db_path, session_id, symbol,
        df["timestamp"].min(), df["timestamp"].max(), len(df)
    )
    logger.debug("Saved %d candles for %s %s", len(df), symbol, timeframe)
