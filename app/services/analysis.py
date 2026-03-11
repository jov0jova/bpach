"""
Phase 7: Profitability Analysis & Reverse-Engineering service.
For every winning trade, extract indicator snapshots, candle context,
and compute statistical patterns.
"""
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import models as m
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)

# Indicators to include in the snapshot (subset for performance)
SNAPSHOT_INDICATORS = [
    "RSI_14", "RSI_7", "RSI_21",
    "MACD", "MACD_signal", "MACD_hist",
    "EMA_20", "EMA_50", "EMA_200",
    "ATR_14", "ADX_14", "DMP_14", "DMN_14",
    "BB_upper_20", "BB_lower_20", "BB_pct_20", "BB_width_20",
    "STOCH_K", "STOCH_D", "STOCHRSI_K",
    "MFI_14", "CCI_20", "WILLR_14", "ROC_10",
    "OBV", "volume_ratio",
    "body_pct", "upper_wick", "lower_wick", "close_pct_change",
    "SUPERT_dir", "ema_20_50_cross", "ema_50_200_cross",
]


def run_analysis(task_id: str, db_path: Path, session_id: str,
                 parquet_dir: Path, run_id: str, timeframes: list) -> None:
    """Background task: extract indicator snapshots for all winning trades."""
    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    winners = m.get_trades(db_path, run_id, winners_only=True)
    if not winners:
        progress(1, 1, "No winning trades found for analysis.")
        return

    progress(0, len(winners), "Analysing winning trades…")
    analyses = []
    primary_tf = timeframes[0] if timeframes else "1h"

    for idx, trade in enumerate(winners):
        progress(idx, len(winners), f"Analysing {trade['symbol']} @ {trade['entry_time']}…")

        symbol = trade["symbol"]
        entry_time = pd.Timestamp(trade["entry_time"])

        # Load data for primary timeframe
        snapshot = {}
        candle_ctx = {}
        pattern_flags = {}

        for tf in timeframes:
            path = parquet_path(parquet_dir, session_id, symbol, tf)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Find the candle at entry
            entry_mask = df["timestamp"] == entry_time
            if not entry_mask.any():
                # Find closest candle
                idx_pos = (df["timestamp"] - entry_time).abs().idxmin()
                entry_row = df.loc[idx_pos]
            else:
                entry_row = df[entry_mask].iloc[0]

            row_idx = df.index[df["timestamp"] == entry_row["timestamp"]][0]

            # Indicator snapshot for this timeframe
            tf_snapshot = {}
            for col in SNAPSHOT_INDICATORS:
                if col in df.columns:
                    val = entry_row.get(col)
                    if pd.notna(val):
                        tf_snapshot[col] = round(float(val), 6)
            snapshot[tf] = tf_snapshot

            # Candle context: ±10 bars around entry
            ctx_start = max(0, row_idx - 10)
            ctx_end = min(len(df), row_idx + 11)
            ctx_df = df.iloc[ctx_start:ctx_end][
                ["timestamp", "open", "high", "low", "close", "volume"]
            ].copy()
            ctx_df["timestamp"] = ctx_df["timestamp"].astype(str)
            candle_ctx[tf] = ctx_df.to_dict("records")

            # Pattern flags at entry
            cdl_cols = [c for c in df.columns if c.startswith("CDL_")]
            for cdl in cdl_cols:
                val = entry_row.get(cdl)
                if pd.notna(val) and val != 0:
                    pattern_flags[f"{tf}_{cdl}"] = int(val)

        analyses.append({
            "run_id": run_id,
            "trade_id": trade["id"],
            "session_id": session_id,
            "symbol": symbol,
            "timeframe": primary_tf,
            "entry_time": entry_time,
            "indicator_snapshot": snapshot,
            "candle_context": candle_ctx,
            "pattern_flags": pattern_flags,
        })

    m.insert_entry_analysis(db_path, analyses)
    progress(len(winners), len(winners),
             f"Analysis complete: {len(analyses)} winning trades reverse-engineered.")
    m.update_task(db_path, task_id, result={"run_id": run_id, "count": len(analyses)})


def compute_indicator_stats(analyses: list, timeframe: str = None) -> dict:
    """
    Given entry analyses, compute per-indicator statistics:
    min, max, mean, std, percentiles.
    Returns dict: { indicator_name: { mean, std, p25, p50, p75, min, max } }
    """
    # Collect values per indicator
    values: dict[str, list] = defaultdict(list)
    for a in analyses:
        snap = a.get("indicator_snapshot", {})
        # If timeframe specified, use that TF's snapshot; else use primary
        if timeframe and timeframe in snap:
            tf_snap = snap[timeframe]
        elif snap:
            tf_snap = list(snap.values())[0]
        else:
            continue
        for k, v in tf_snap.items():
            if v is not None:
                values[k].append(float(v))

    stats = {}
    for ind, vals in values.items():
        if len(vals) < 3:
            continue
        arr = np.array(vals)
        stats[ind] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "count": len(vals),
        }
    return stats


def compute_win_rate_by_condition(analyses: list, all_trades: list,
                                  timeframe: str = None) -> dict:
    """
    For each indicator, check if having that indicator in a certain range
    correlates with wins. Returns { condition_label: win_rate }.
    """
    # Simple version: bucket RSI/ADX into ranges and compute win rate
    conditions = {
        "RSI_14 < 30 (Oversold)": lambda s: s.get("RSI_14", 50) < 30,
        "RSI_14 30-50": lambda s: 30 <= s.get("RSI_14", 50) < 50,
        "RSI_14 50-70": lambda s: 50 <= s.get("RSI_14", 50) < 70,
        "ADX_14 > 25 (Strong trend)": lambda s: s.get("ADX_14", 0) > 25,
        "MACD_hist > 0 (Bullish)": lambda s: s.get("MACD_hist", 0) > 0,
        "BB_pct_20 < 0.2 (Near lower band)": lambda s: s.get("BB_pct_20", 0.5) < 0.2,
        "volume_ratio > 1.5 (High vol)": lambda s: s.get("volume_ratio", 1) > 1.5,
        "close > EMA_50": lambda s: s.get("EMA_50") and s.get("close", 0) > s.get("EMA_50", 0),
    }

    def get_snap(a):
        snap = a.get("indicator_snapshot", {})
        if timeframe and timeframe in snap:
            return snap[timeframe]
        elif snap:
            return list(snap.values())[0]
        return {}

    results = {}
    for label, fn in conditions.items():
        matching = [a for a in analyses if fn(get_snap(a))]
        if len(matching) >= 5:
            # All these are winning trades, so win rate is 100% for the analysis subset
            # Compare to total win rate to get relative strength
            results[label] = len(matching) / len(analyses) if analyses else 0

    return results


def compute_correlation_matrix(analyses: list, timeframe: str = None) -> dict:
    """
    Compute indicator correlations among winning trade entries.
    Returns { ind_a: { ind_b: correlation } }.
    """
    # Build DataFrame from snapshots
    rows = []
    for a in analyses:
        snap = a.get("indicator_snapshot", {})
        if timeframe and timeframe in snap:
            tf_snap = snap[timeframe]
        elif snap:
            tf_snap = list(snap.values())[0]
        else:
            continue
        rows.append(tf_snap)

    if len(rows) < 5:
        return {}

    df = pd.DataFrame(rows)
    # Keep only numeric columns with enough non-null values
    num_cols = [c for c in df.columns
                if df[c].notna().sum() >= len(df) * 0.7 and
                df[c].dtype in (float, int, "float64", "int64")]
    df = df[num_cols].dropna(axis=1, thresh=int(len(df) * 0.7))

    if df.empty or df.shape[1] < 2:
        return {}

    corr = df.corr()
    result = {}
    for col in corr.columns:
        result[col] = {c: round(float(corr.loc[col, c]), 3) for c in corr.columns}
    return result
