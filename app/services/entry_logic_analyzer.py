"""
Path A: Entry Logic Analyzer.

Given a user-defined entry condition (Python expression using indicator column names),
this service:
1. Evaluates the condition on all historical candles to find entry points.
2. Simulates holding each entry for `hold_bars` candles, classifying each as
   a winner (return > min_profit_pct) or loser.
3. Computes Cohen's D for every indicator between winners and losers.
4. Derives suggested threshold ranges from the winner distribution.
5. Returns the top discriminative indicators with recommended filter rules.
"""
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy, inject_htf_features, HTF_INJECT_COLS
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)

# Indicators to compare between winners and losers
ANALYSIS_INDICATORS = [
    "RSI_14", "RSI_7", "RSI_21",
    "MACD", "MACD_signal", "MACD_hist",
    "EMA_8", "EMA_20", "EMA_50", "EMA_200",
    "ATR_14", "ADX_14",
    "BB_pct_20", "BB_width_20",
    "STOCH_K", "STOCH_D", "STOCHRSI_K",
    "MFI_14", "WILLR_14", "ROC_10", "CMF_20",
    "volume_ratio",
    "body_pct", "upper_wick", "lower_wick", "close_pct_change",
    "ema_20_50_cross", "ema_50_200_cross",
    "AROON_up", "AROON_down", "AO",
    "SUPERT_dir",
]


def _parse_direction(entry_logic: str) -> tuple[str, str]:
    """Extract direction comment and return clean code."""
    direction = "long"
    lines = entry_logic.strip().splitlines()
    clean = []
    for line in lines:
        m = re.match(r"#\s*direction:\s*(\w+)", line.strip())
        if m:
            direction = m.group(1).lower()
        else:
            clean.append(line)
    return direction, "\n".join(clean).strip()


def _eval_entry_condition(df: pd.DataFrame, condition_code: str) -> pd.Series:
    """
    Evaluate a Python boolean expression against a DataFrame row-by-row.
    The expression can reference any column name directly.
    Returns a boolean Series.
    """
    # Build a namespace from the DataFrame columns
    ns = {col: df[col] for col in df.columns if col.isidentifier()}
    ns.update({"pd": pd, "np": np})
    try:
        result = eval(condition_code, {"__builtins__": {}}, ns)  # noqa: S307
        if isinstance(result, pd.Series):
            return result.fillna(False).astype(bool)
        return pd.Series(bool(result), index=df.index)
    except Exception as e:
        logger.warning("Entry condition eval failed: %s", e)
        return pd.Series(False, index=df.index)


def _simulate_entries(df: pd.DataFrame, entry_mask: pd.Series,
                      hold_bars: int, min_profit_pct: float,
                      fee_rate: float = 0.001, slippage: float = 0.0005,
                      analysis_cols: list | None = None) -> list[dict]:
    """
    For each entry signal, simulate holding for hold_bars candles.
    Returns list of dicts: {entry_idx, is_winner, pnl_pct, indicator_snapshot}
    """
    if analysis_cols is None:
        analysis_cols = ANALYSIS_INDICATORS

    close = df["close"].values
    entries = []
    entry_indices = entry_mask[entry_mask].index.tolist()

    for idx in entry_indices:
        i = df.index.get_loc(idx)
        if i + hold_bars >= len(df):
            continue
        entry_price = close[i] * (1 + slippage)
        exit_price = close[i + hold_bars] * (1 - slippage)
        pnl_pct = (exit_price / entry_price - 1) * 100 - fee_rate * 200

        # Snapshot indicator values at entry (primary + HTF)
        snap = {}
        for col in analysis_cols:
            if col in df.columns:
                val = df[col].iloc[i]
                if pd.notna(val):
                    snap[col] = float(val)

        entries.append({
            "entry_idx": i,
            "pnl_pct": pnl_pct,
            "is_winner": pnl_pct > min_profit_pct,
            "snapshot": snap,
        })

    return entries


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's D = (mean_a - mean_b) / pooled_std"""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _suggest_filter(col: str, winner_vals: np.ndarray, loser_vals: np.ndarray,
                    cohens_d_val: float) -> Optional[dict]:
    """
    Given winner and loser distributions for one indicator, suggest a filter rule.
    Returns: {col, operator, threshold, description} or None.
    """
    w_mean = float(np.mean(winner_vals))
    l_mean = float(np.mean(loser_vals))
    w_p25 = float(np.percentile(winner_vals, 25))
    w_p75 = float(np.percentile(winner_vals, 75))

    # Determine direction: are winners consistently higher or lower?
    if cohens_d_val > 0:
        # Winners have higher values → use threshold above which winners cluster
        operator = ">"
        threshold = round(w_p25, 4)
        description = f"{col} > {threshold:.4g}  (winners avg {w_mean:.4g}, losers avg {l_mean:.4g})"
    else:
        # Winners have lower values
        operator = "<"
        threshold = round(w_p75, 4)
        description = f"{col} < {threshold:.4g}  (winners avg {w_mean:.4g}, losers avg {l_mean:.4g})"

    return {
        "col": col,
        "operator": operator,
        "threshold": threshold,
        "winner_mean": round(w_mean, 4),
        "loser_mean": round(l_mean, 4),
        "winner_p25": round(w_p25, 4),
        "winner_p75": round(w_p75, 4),
        "cohens_d": round(cohens_d_val, 3),
        "abs_d": round(abs(cohens_d_val), 3),
        "description": description,
    }


def _ml_feature_importance(winners: list[dict], losers: list[dict],
                            top_n: int = 15) -> list[dict]:
    """
    Random Forest feature importance for winner/loser classification.
    Captures nonlinear relationships and interaction effects that Cohen's D misses.
    Returns list of {col, rf_importance, rank} sorted by importance descending.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        logger.warning("scikit-learn not available for RF feature importance.")
        return []

    # Build feature matrix
    all_samples = winners + losers
    labels = [1] * len(winners) + [0] * len(losers)

    snaps = [e["snapshot"] for e in all_samples]
    if not snaps:
        return []

    df = pd.DataFrame(snaps).fillna(0)
    if df.shape[1] < 2 or df.shape[0] < 20:
        return []

    # Keep only columns that exist in most rows
    df = df.dropna(axis=1, thresh=int(len(df) * 0.6))
    cols = df.columns.tolist()

    X = df.values
    y = np.array(labels)

    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)

        importances = rf.feature_importances_
        result = sorted(
            [{"col": col, "rf_importance": round(float(imp), 5)}
             for col, imp in zip(cols, importances)
             if imp > 0.001],
            key=lambda x: x["rf_importance"], reverse=True
        )
        for i, r in enumerate(result[:top_n]):
            r["rank"] = i + 1
        return result[:top_n]
    except Exception as e:
        logger.warning("RF feature importance failed: %s", e)
        return []


def run_entry_analysis(task_id: str, db_path: Path, session_id: str,
                       parquet_dir: Path, timeframes: list,
                       entry_logic: str, hold_bars: int = 10,
                       min_profit_pct: float = 0.5) -> None:
    """
    Background task: evaluate entry logic on all pairs, classify winners/losers,
    run Cohen's D analysis, and store top discriminative indicators.
    """
    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    direction, condition_code = _parse_direction(entry_logic)
    if not condition_code:
        m.update_task(db_path, task_id, status="error",
                      error="Empty entry condition.",
                      message="Please provide a valid entry condition.")
        return

    pairs = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    primary_tf = timeframes[0] if timeframes else "1h"

    if not active:
        m.update_task(db_path, task_id, status="error",
                      error="No active pairs with data.",
                      message="Download data and add indicators first.")
        return

    higher_tfs = timeframes[1:] if len(timeframes) > 1 else []

    progress(0, len(active), f"Evaluating entry logic on {len(active)} pairs…")

    base_strategy = BaseStrategy()
    all_entries: list[dict] = []
    pairs_processed = 0
    htf_cols_available: set = set()

    for pair in active:
        path = parquet_path(parquet_dir, session_id, pair["symbol"], primary_tf)
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("Cannot read %s: %s", path, e)
            continue

        if len(df) < 100:
            continue

        # Ensure indicators are present; if not, compute them
        if "RSI_14" not in df.columns:
            df = base_strategy.populate_indicators(df)

        # Inject HTF features so entry conditions can reference HTF_ columns
        for htf in higher_tfs:
            htf_path = parquet_path(parquet_dir, session_id, pair["symbol"], htf)
            if not htf_path.exists():
                continue
            try:
                htf_df = pd.read_parquet(htf_path)
                df = inject_htf_features(df, htf_df, htf, base_strategy)
                # Track which HTF columns are available for analysis
                for c in df.columns:
                    if c.startswith(f"HTF_{htf}_"):
                        htf_cols_available.add(c)
            except Exception as e:
                logger.warning("Entry analyzer HTF inject %s %s: %s", pair["symbol"], htf, e)

        entry_mask = _eval_entry_condition(df, condition_code)

        # Build full indicator list including any HTF columns present in this df
        htf_cols_in_df = [c for c in df.columns if c.startswith("HTF_")]
        full_analysis_cols = ANALYSIS_INDICATORS + htf_cols_in_df
        entries = _simulate_entries(df, entry_mask, hold_bars, min_profit_pct,
                                    analysis_cols=full_analysis_cols)
        all_entries.extend(entries)
        pairs_processed += 1
        progress(pairs_processed, len(active),
                 f"Processed {pairs_processed}/{len(active)} pairs — {len(all_entries)} entries found")

    if len(all_entries) < 10:
        m.update_task(db_path, task_id, status="error",
                      error=f"Only {len(all_entries)} entries found — need at least 10.",
                      message="Entry condition may be too restrictive, or data not enriched yet.")
        return

    winners = [e for e in all_entries if e["is_winner"]]
    losers = [e for e in all_entries if not e["is_winner"]]

    progress(len(active), len(active),
             f"Comparing {len(winners)} winners vs {len(losers)} losers…")

    if len(winners) < 5 or len(losers) < 5:
        m.update_task(db_path, task_id, status="error",
                      error=f"Need ≥5 winners and ≥5 losers. Got {len(winners)} / {len(losers)}.",
                      message="Try a longer hold period or lower min_profit threshold.")
        return

    # Build DataFrames for winners/losers indicator snapshots
    w_snaps = pd.DataFrame([e["snapshot"] for e in winners])
    l_snaps = pd.DataFrame([e["snapshot"] for e in losers])

    # Compute Cohen's D for each available indicator
    discrimination = []
    for col in ANALYSIS_INDICATORS:
        if col not in w_snaps.columns or col not in l_snaps.columns:
            continue
        w_vals = w_snaps[col].dropna().values
        l_vals = l_snaps[col].dropna().values
        if len(w_vals) < 5 or len(l_vals) < 5:
            continue
        d = _cohens_d(w_vals, l_vals)
        if abs(d) < 0.1:
            continue
        suggestion = _suggest_filter(col, w_vals, l_vals, d)
        if suggestion:
            discrimination.append(suggestion)

    discrimination.sort(key=lambda x: x["abs_d"], reverse=True)
    top_indicators = discrimination[:15]

    # ML Feature Importance
    rf_importance = _ml_feature_importance(winners, losers)

    # Merge RF rank into top_indicators
    rf_map = {r["col"]: r["rf_importance"] for r in rf_importance}
    for ind in top_indicators:
        ind["rf_importance"] = round(rf_map.get(ind["col"], 0.0), 5)

    # Combined score: average of normalized Cohen's D and RF importance
    max_d  = max((i["abs_d"] for i in top_indicators), default=1.0) or 1.0
    max_rf = max((i["rf_importance"] for i in top_indicators), default=1.0) or 1.0
    for ind in top_indicators:
        norm_d  = ind["abs_d"] / max_d
        norm_rf = ind["rf_importance"] / max_rf
        ind["combined_score"] = round((norm_d + norm_rf) / 2, 4)
    top_indicators.sort(key=lambda x: x["combined_score"], reverse=True)

    # Overall stats
    win_rate = len(winners) / len(all_entries) if all_entries else 0
    avg_winner_pnl = float(np.mean([e["pnl_pct"] for e in winners])) if winners else 0
    avg_loser_pnl = float(np.mean([e["pnl_pct"] for e in losers])) if losers else 0

    result = {
        "total_entries": len(all_entries),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 4),
        "avg_winner_pnl": round(avg_winner_pnl, 4),
        "avg_loser_pnl": round(avg_loser_pnl, 4),
        "hold_bars": hold_bars,
        "min_profit_pct": min_profit_pct,
        "direction": direction,
        "entry_condition": condition_code,
        "top_indicators": top_indicators,
        "rf_importance": rf_importance,
        "pairs_processed": pairs_processed,
    }

    m.update_task(db_path, task_id,
                  status="done",
                  progress=len(active),
                  total=len(active),
                  message=f"Done — {len(winners)}/{len(all_entries)} winners, {len(top_indicators)} discriminative indicators found.",
                  result=result)
