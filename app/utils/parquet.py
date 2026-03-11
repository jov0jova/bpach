"""
Parquet file helpers: read/write OHLCV + indicator data.
Files are stored as: data/parquet/{session_id}/{symbol_safe}/{timeframe}.parquet
"""
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _symbol_dir(parquet_dir: Path, session_id: str, symbol: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", symbol)
    return parquet_dir / session_id / safe


def parquet_path(parquet_dir: Path, session_id: str, symbol: str, timeframe: str) -> Path:
    return _symbol_dir(parquet_dir, session_id, symbol) / f"{timeframe}.parquet"


def write_ohlcv(parquet_dir: Path, session_id: str, symbol: str, timeframe: str,
                df: pd.DataFrame) -> None:
    """Write (or merge) OHLCV DataFrame to parquet. Deduplicates on timestamp."""
    path = parquet_path(parquet_dir, session_id, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df.reset_index(drop=True).to_parquet(path, index=False)


def read_ohlcv(parquet_dir: Path, session_id: str, symbol: str, timeframe: str,
               start=None, end=None) -> pd.DataFrame | None:
    """Read OHLCV parquet. Returns None if file doesn't exist."""
    path = parquet_path(parquet_dir, session_id, symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def list_available_symbols(parquet_dir: Path, session_id: str, timeframe: str) -> list:
    """Return list of symbols that have parquet data for the given timeframe."""
    base = parquet_dir / session_id
    if not base.exists():
        return []
    symbols = []
    for sym_dir in base.iterdir():
        if (sym_dir / f"{timeframe}.parquet").exists():
            # Reverse the safe name substitution (best effort)
            symbols.append(sym_dir.name.replace("_", "/", 1))
    return sorted(symbols)


def write_enriched(parquet_dir: Path, session_id: str, symbol: str, timeframe: str,
                   df: pd.DataFrame) -> None:
    """Write enriched DataFrame (OHLCV + indicators) back to parquet."""
    path = parquet_path(parquet_dir, session_id, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)
