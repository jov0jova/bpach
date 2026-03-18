"""
Information Coefficient (IC) Analysis Service.

Three levels of analysis
─────────────────────────
1. Raw indicator IC  — Spearman rank correlation between a raw indicator value
   (e.g. RSI_14 = 42.3) and the forward return N bars later.
   Answers: "does knowing the *value* of this indicator predict price?"

2. Condition IC  — IC for named binary trading conditions
   (e.g. "RSI < 35" fires True/False each bar).
   Answers: "when this condition is TRUE, are forward returns higher?"
   Also reports hit rate — how often the condition fires.

3. Combination IC  — IC for AND-pairs of the best conditions
   (e.g. "RSI < 35 AND EMA(20) > EMA(50)").
   Answers: "does combining two conditions give a better signal than either alone?"
   Synergy = combo_IC / max(ic_a, ic_b) — values > 1 mean genuine combination edge.

IC values:
  |IC| > 0.05  — weak but meaningful (worth testing)
  |IC| > 0.10  — moderate predictive power
  |IC| > 0.20  — strong (rare in practice)

ICIR (IC Information Ratio) = mean(IC) / std(IC)
  ICIR > 0.5  — consistent signal
  ICIR > 1.0  — strong, reliable signal
"""
import logging
from itertools import combinations as iter_combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .. import models as m
from ..strategies.base import BaseStrategy, inject_htf_features, HTF_INJECT_COLS
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)

FORWARD_PERIODS = [1, 5, 10, 20, 40]  # bars ahead

# Columns to always skip — raw OHLCV, metadata, derived signals
_IC_SKIP_COLS = frozenset({
    "open", "high", "low", "close", "volume",
    "timestamp", "date", "symbol",
    "entry_signal", "exit_signal",
    "MACD", "MACD_signal",           # raw line values; MACD_hist is what matters
    "STOCH_D", "STOCHRSI_D",
    "PSAR_up", "PSAR_down",
    "BB_upper_14", "BB_mid_14", "BB_upper_20", "BB_mid_20",
    "KC_upper", "KC_middle", "DC_upper", "DC_middle",
})


def _auto_detect_cols(df: pd.DataFrame) -> list:
    """
    Return all numeric/boolean columns worth running IC on.
    Skips raw OHLCV, metadata, near-constant columns, and all-NaN columns.
    This is better than a hardcoded list — it adapts to whatever indicators
    were actually computed on this dataset.
    """
    cols = []
    for col in df.columns:
        if col in _IC_SKIP_COLS:
            continue
        # Skip HTF raw OHLCV
        if any(col.endswith(f"_{s}") for s in ("open", "high", "low", "close", "volume")):
            continue
        # Skip string/object columns
        if df[col].dtype == object:
            continue
        # Skip all-NaN
        if df[col].isna().all():
            continue
        # Skip near-constant (< 2 distinct values after dropping NaN)
        if df[col].dropna().nunique() < 2:
            continue
        cols.append(col)
    return cols


# ── Named binary conditions (aligned with strategy slot options) ─────────────
# Each entry: (label, col_a, op, col_b_or_scalar)
# op: "<" | ">" | "<=" | ">=" | "=="
# col_b_or_scalar: column name (str) or numeric value

SLOT_CONDITIONS = [
    # Trend state conditions
    ("EMA(20) > EMA(50)",     "EMA_20",   ">",  "EMA_50"),
    ("EMA(50) > EMA(200)",    "EMA_50",   ">",  "EMA_200"),
    ("Price > EMA(50)",       "close",    ">",  "EMA_50"),
    ("Price > EMA(200)",      "close",    ">",  "EMA_200"),
    ("Supertrend Bullish",    "SUPERT_dir", "==", 1),
    ("PSAR Bullish",          "PSAR_dir",  "==", 1),
    # Oscillator oversold conditions
    ("RSI < 30",              "RSI_14",    "<",  30),
    ("RSI < 35",              "RSI_14",    "<",  35),
    ("RSI < 40",              "RSI_14",    "<",  40),
    ("RSI > 50",              "RSI_14",    ">",  50),
    ("RSI > 60",              "RSI_14",    ">",  60),
    ("Stoch %K < 20",         "STOCH_K",   "<",  20),
    ("Stoch %K < 30",         "STOCH_K",   "<",  30),
    ("StochRSI < 20",         "STOCHRSI_K","<",  20),
    ("CCI < -100",            "CCI_20",    "<",  -100),
    ("CCI < -150",            "CCI_20",    "<",  -150),
    ("MFI < 25",              "MFI_14",    "<",  25),
    ("MFI < 35",              "MFI_14",    "<",  35),
    ("Williams %R < -70",     "WILLR_14",  "<",  -70),
    ("MACD hist > 0",         "MACD_hist", ">",  0),
    ("Price below BB lower",  "close",     "<",  "BB_lower_20"),
    ("ADX > 20",              "ADX_14",    ">",  20),
    ("ADX > 25",              "ADX_14",    ">",  25),
    ("Volume > 1.5×avg",      "volume_ratio", ">", 1.5),
    ("CMF > 0",               "CMF_20",    ">",  0),
    ("Price above BB upper",  "close",     ">",  "BB_upper_20"),
]


def _eval_condition(df: pd.DataFrame, spec: tuple) -> "pd.Series | None":
    """Evaluate a SLOT_CONDITIONS entry → boolean Series or None if col missing."""
    _, col_a, op, col_b = spec
    if col_a not in df.columns:
        return None
    a = df[col_a]
    if isinstance(col_b, str):
        if col_b not in df.columns:
            return None
        b = df[col_b]
    else:
        b = col_b
    if op == "<":  return a < b
    if op == ">":  return a > b
    if op == "<=": return a <= b
    if op == ">=": return a >= b
    if op == "==": return a == b
    return None


# ── IC computation ────────────────────────────────────────────────────────────

def _ic_for_series(series: pd.Series, forward_returns: dict,
                   min_obs: int = 30) -> dict:
    """
    Compute IC (Spearman rank correlation) between a series (raw or binary)
    and forward returns at multiple horizons.
    """
    result = {}
    ic_values_by_period = {}

    for period, fwd_ret in forward_returns.items():
        aligned = pd.concat([series, fwd_ret], axis=1).dropna()
        if len(aligned) < min_obs:
            result[f"ic_{period}"] = float("nan")
            continue
        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values
        try:
            rho, pval = scipy_stats.spearmanr(x, y)
            result[f"ic_{period}"] = round(float(rho), 4)
            result[f"pval_{period}"] = round(float(pval), 4)
            ic_values_by_period[period] = rho
        except Exception:
            result[f"ic_{period}"] = float("nan")

    # Rolling IC stability at the primary horizon
    primary_period = 10
    if primary_period in forward_returns:
        fwd = forward_returns[primary_period]
        aligned = pd.concat([series, fwd], axis=1).dropna()
        if len(aligned) >= 60:
            window = max(30, len(aligned) // 5)
            x_rank = aligned.iloc[:, 0].rank()
            y_rank = aligned.iloc[:, 1].rank()
            with np.errstate(invalid="ignore", divide="ignore"):
                rolling_corr = x_rank.rolling(window=window,
                                              min_periods=max(20, window // 2)).corr(y_rank)
            step = max(1, len(aligned) // 20)
            roll_ics = [v for v in rolling_corr.iloc[::step].tolist() if not np.isnan(v)]
            if len(roll_ics) >= 3:
                ic_mean = float(np.nanmean(roll_ics))
                ic_std  = float(np.nanstd(roll_ics))
                result["ic_mean"]         = round(ic_mean, 4)
                result["ic_std"]          = round(ic_std, 4)
                result["icir"]            = round(ic_mean / ic_std, 3) if ic_std > 1e-6 else 0.0
                result["ic_positive_pct"] = round(
                    sum(1 for v in roll_ics if v > 0) / len(roll_ics), 3)

    valid_ics = {k: v for k, v in ic_values_by_period.items() if not np.isnan(v)}
    if valid_ics:
        result["best_period"] = max(valid_ics, key=lambda k: abs(valid_ics[k]))
        result["best_ic"]     = round(valid_ics[result["best_period"]], 4)
        result["max_abs_ic"]  = round(abs(result["best_ic"]), 4)
        result["direction"]   = "bullish" if result["best_ic"] > 0 else "bearish"
    return result


def _regime_split(df: pd.DataFrame) -> dict:
    regimes = {}
    if "EMA50_slope" in df.columns:
        slope = df["EMA50_slope"]
        regimes["trending_up"]   = df[slope > 0.05]
        regimes["trending_down"] = df[slope < -0.05]
        regimes["ranging"]       = df[slope.abs() <= 0.05]
    if "NATR_14" in df.columns:
        natr = df["NATR_14"].dropna()
        q25, q75 = natr.quantile(0.25), natr.quantile(0.75)
        regimes["high_vol"] = df[df["NATR_14"] >= q75]
        regimes["low_vol"]  = df[df["NATR_14"] <= q25]
    return regimes


def _build_htf_analysis_cols(timeframes: list) -> list:
    htf_cols = []
    for tf in timeframes[1:]:
        for base_col in HTF_INJECT_COLS:
            htf_cols.append(f"HTF_{tf}_{base_col}")
        htf_cols.append(f"HTF_{tf}_trend_dir")
        htf_cols.append(f"HTF_{tf}_regime")
    return htf_cols


# ── Main analysis runner ──────────────────────────────────────────────────────

def run_ic_analysis(task_id: str, db_path: Path, session_id: str,
                    parquet_dir: Path, timeframes: list, stop_event=None) -> None:
    """
    Background task: compute IC for every indicator across all pairs.

    Three passes per pair:
      1. Raw indicator IC  — all numeric columns auto-detected from df
      2. Condition IC      — named binary conditions from SLOT_CONDITIONS
      3. Combination IC    — AND-pairs of the top conditions (per-pair)

    All results are aggregated (mean across pairs) before saving.
    """
    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    pairs      = m.list_pairs(db_path, session_id)
    active     = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    primary_tf = timeframes[0] if timeframes else "1h"
    higher_tfs = timeframes[1:] if len(timeframes) > 1 else []

    if not active:
        m.update_task(db_path, task_id, status="error",
                      error="No active pairs with data.",
                      message="Download data and add indicators first.")
        return

    sample = active[:30]
    tf_desc = primary_tf + (f" + HTF: {', '.join(higher_tfs)}" if higher_tfs else "")
    progress(0, len(sample), f"Computing IC for {len(sample)} pairs [{tf_desc}]…")

    base_strategy = BaseStrategy()

    # Accumulators: {indicator_col: {period: [ic_values...]}}
    pair_ics:      dict[str, dict]       = {}
    regime_ics:    dict[str, dict]       = {}

    # Condition IC: {label: {ic_10: [...], hit_rate: [...]}}
    cond_ics:      dict[str, dict]       = {}

    # Combination IC: {(label_a, label_b): {ic_10: [...], hit_rate: [...]}}
    combo_ics:     dict[tuple, dict]     = {}

    pairs_done = 0

    for pair in sample:
        progress(pairs_done, len(sample),
                 f"Pair {pairs_done + 1}/{len(sample)}: {pair['symbol']}…")

        path = parquet_path(parquet_dir, session_id, pair["symbol"], primary_tf)
        if not path.exists():
            pairs_done += 1
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("IC analysis: cannot read %s: %s", path, e)
            pairs_done += 1
            continue

        if len(df) < 200:
            pairs_done += 1
            continue

        if "RSI_14" not in df.columns or df["RSI_14"].isna().all():
            df = base_strategy.populate_indicators(df)

        for htf in higher_tfs:
            htf_path = parquet_path(parquet_dir, session_id, pair["symbol"], htf)
            if not htf_path.exists():
                continue
            try:
                htf_df = pd.read_parquet(htf_path)
                df = inject_htf_features(df, htf_df, htf, base_strategy)
            except Exception as e:
                logger.warning("IC: HTF inject failed %s %s: %s", pair["symbol"], htf, e)

        # Forward returns
        fwd_returns: dict[int, pd.Series] = {
            p: df["close"].pct_change(p).shift(-p)
            for p in FORWARD_PERIODS
        }
        fwd_10 = fwd_returns.get(10)

        # ── Pass 1: Raw indicator IC (all numeric columns) ────────────────────
        analysis_cols = _auto_detect_cols(df)
        regimes       = _regime_split(df)

        for col in analysis_cols:
            series = df[col]
            ic_data = _ic_for_series(series, fwd_returns)

            if col not in pair_ics:
                pair_ics[col] = {f"ic_{p}": [] for p in FORWARD_PERIODS}
                pair_ics[col]["icir_list"]             = []
                pair_ics[col]["ic_positive_pct_list"]  = []

            for p in FORWARD_PERIODS:
                val = ic_data.get(f"ic_{p}")
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    pair_ics[col][f"ic_{p}"].append(val)

            if "icir" in ic_data and not np.isnan(ic_data["icir"]):
                pair_ics[col]["icir_list"].append(ic_data["icir"])
            if "ic_positive_pct" in ic_data:
                pair_ics[col]["ic_positive_pct_list"].append(ic_data["ic_positive_pct"])

            for regime_name, regime_df in regimes.items():
                if len(regime_df) < 50 or col not in regime_df.columns:
                    continue
                r_fwd = {10: regime_df["close"].pct_change(10).shift(-10)}
                r_ic  = _ic_for_series(regime_df[col], r_fwd, min_obs=20)
                r_val = r_ic.get("ic_10")
                if r_val is not None and not (isinstance(r_val, float) and np.isnan(r_val)):
                    regime_ics.setdefault(col, {}).setdefault(regime_name, []).append(r_val)

        if fwd_10 is None:
            pairs_done += 1
            continue

        # ── Pass 2: Condition IC ──────────────────────────────────────────────
        # Evaluate each named binary condition; compute IC at 10-bar horizon
        active_conds: list[tuple[str, pd.Series]] = []   # (label, bool series)
        for spec in SLOT_CONDITIONS:
            label = spec[0]
            cond  = _eval_condition(df, spec)
            if cond is None:
                continue
            hit_rate = float(cond.mean())
            # Skip if condition fires on <0.5% or >99% of bars — not useful
            if hit_rate < 0.005 or hit_rate > 0.99:
                continue

            ic_data = _ic_for_series(cond.astype(float), {10: fwd_10}, min_obs=30)
            ic_val  = ic_data.get("ic_10")
            if ic_val is None or (isinstance(ic_val, float) and np.isnan(ic_val)):
                continue

            if label not in cond_ics:
                cond_ics[label] = {"ic_10": [], "hit_rate": []}
            cond_ics[label]["ic_10"].append(ic_val)
            cond_ics[label]["hit_rate"].append(hit_rate)
            active_conds.append((label, cond))

        # ── Pass 3: Combination IC (AND-pairs of active conditions) ───────────
        # Limit to top 15 conditions to keep pairs count manageable (15*14/2=105)
        # Use the raw IC value from this pair to pick top-15 for this pair
        top_conds = sorted(
            active_conds,
            key=lambda t: abs(cond_ics[t[0]]["ic_10"][-1]) if cond_ics[t[0]]["ic_10"] else 0,
            reverse=True,
        )[:15]

        for (label_a, cond_a), (label_b, cond_b) in iter_combinations(top_conds, 2):
            combo     = (cond_a & cond_b).astype(float)
            hit_rate  = float(combo.mean())
            if hit_rate < 0.005:   # fires on < 0.5% of bars → unreliable
                continue
            ic_data = _ic_for_series(combo, {10: fwd_10}, min_obs=20)
            ic_val  = ic_data.get("ic_10")
            if ic_val is None or (isinstance(ic_val, float) and np.isnan(ic_val)):
                continue

            key = (label_a, label_b)
            if key not in combo_ics:
                combo_ics[key] = {"ic_10": [], "hit_rate": [],
                                  "ic_a": [], "ic_b": []}
            combo_ics[key]["ic_10"].append(ic_val)
            combo_ics[key]["hit_rate"].append(hit_rate)
            # Store individual ICs from this pair for synergy calculation
            ic_a = cond_ics[label_a]["ic_10"][-1] if cond_ics[label_a]["ic_10"] else 0
            ic_b = cond_ics[label_b]["ic_10"][-1] if cond_ics[label_b]["ic_10"] else 0
            combo_ics[key]["ic_a"].append(ic_a)
            combo_ics[key]["ic_b"].append(ic_b)

        pairs_done += 1

    if not pair_ics:
        m.update_task(db_path, task_id, status="error",
                      error="No indicators found. Run Add Indicators first.",
                      message="Run Phase 5 (Add Indicators) before IC analysis.")
        return

    progress(len(sample), len(sample), "Aggregating results…")

    # ── Aggregate raw indicator IC ────────────────────────────────────────────
    results = []
    for col, period_data in pair_ics.items():
        row = {"col": col}
        valid_ics = {}
        for period in FORWARD_PERIODS:
            vals = period_data.get(f"ic_{period}", [])
            if vals:
                avg = float(np.nanmean(vals))
                row[f"ic_{period}"] = round(avg, 4)
                valid_ics[period]   = avg
            else:
                row[f"ic_{period}"] = None
        if not valid_ics:
            continue

        icir_vals = period_data.get("icir_list", [])
        row["icir"] = round(float(np.nanmean(icir_vals)), 3) if icir_vals else None

        pos_pct = period_data.get("ic_positive_pct_list", [])
        row["ic_positive_pct"] = round(float(np.nanmean(pos_pct)), 3) if pos_pct else None

        abs_ics  = {p: abs(v) for p, v in valid_ics.items()}
        best_p   = max(abs_ics, key=abs_ics.get)
        row["best_period"] = best_p
        row["best_ic"]     = round(valid_ics[best_p], 4)
        row["max_abs_ic"]  = round(abs(valid_ics[best_p]), 4)
        row["direction"]   = "bullish" if valid_ics[best_p] > 0 else "bearish"

        if col in regime_ics:
            row["regime_ic"] = {
                regime: round(float(np.nanmean(vals)), 4)
                for regime, vals in regime_ics[col].items() if vals
            }
            if row["regime_ic"]:
                best_r = max(row["regime_ic"], key=lambda k: abs(row["regime_ic"][k]))
                row["best_regime"]    = best_r
                row["best_regime_ic"] = row["regime_ic"][best_r]
        else:
            row["regime_ic"] = {}

        max_abs = row["max_abs_ic"]
        row["strength"] = (
            "strong"   if max_abs >= 0.20 else
            "moderate" if max_abs >= 0.10 else
            "weak"     if max_abs >= 0.05 else
            "noise"
        )
        results.append(row)

    results.sort(key=lambda r: r["max_abs_ic"], reverse=True)

    # ── Aggregate condition IC ────────────────────────────────────────────────
    cond_results = []
    for label, data in cond_ics.items():
        ic_vals   = data.get("ic_10", [])
        hr_vals   = data.get("hit_rate", [])
        if not ic_vals:
            continue
        avg_ic   = float(np.nanmean(ic_vals))
        avg_hr   = float(np.nanmean(hr_vals))
        cond_results.append({
            "label":     label,
            "ic_10":     round(avg_ic, 4),
            "abs_ic":    round(abs(avg_ic), 4),
            "hit_rate":  round(avg_hr, 4),
            "direction": "bullish" if avg_ic > 0 else "bearish",
            "strength": (
                "strong"   if abs(avg_ic) >= 0.20 else
                "moderate" if abs(avg_ic) >= 0.10 else
                "weak"     if abs(avg_ic) >= 0.05 else
                "noise"
            ),
            "n_pairs": len(ic_vals),
        })
    cond_results.sort(key=lambda r: r["abs_ic"], reverse=True)

    # ── Aggregate combination IC ──────────────────────────────────────────────
    combo_results = []
    for (label_a, label_b), data in combo_ics.items():
        ic_vals  = data.get("ic_10", [])
        hr_vals  = data.get("hit_rate", [])
        ica_vals = data.get("ic_a", [])
        icb_vals = data.get("ic_b", [])
        if len(ic_vals) < 2:   # need at least 2 pairs for a reliable estimate
            continue
        avg_ic   = float(np.nanmean(ic_vals))
        avg_hr   = float(np.nanmean(hr_vals))
        avg_ica  = float(np.nanmean(ica_vals))
        avg_icb  = float(np.nanmean(icb_vals))
        best_ind = max(abs(avg_ica), abs(avg_icb))
        synergy  = abs(avg_ic) / best_ind if best_ind > 1e-6 else 0.0
        combo_results.append({
            "label":     f"{label_a}  AND  {label_b}",
            "label_a":   label_a,
            "label_b":   label_b,
            "ic_10":     round(avg_ic,  4),
            "abs_ic":    round(abs(avg_ic), 4),
            "ic_a":      round(avg_ica, 4),
            "ic_b":      round(avg_icb, 4),
            "synergy":   round(synergy, 3),
            "hit_rate":  round(avg_hr,  4),
            "direction": "bullish" if avg_ic > 0 else "bearish",
            "strength": (
                "strong"   if abs(avg_ic) >= 0.20 else
                "moderate" if abs(avg_ic) >= 0.10 else
                "weak"     if abs(avg_ic) >= 0.05 else
                "noise"
            ),
            "n_pairs": len(ic_vals),
            # Flag combinations where AND is strictly better than both parts alone
            "synergistic": synergy > 1.05 and abs(avg_ic) > 0.03,
        })
    combo_results.sort(key=lambda r: r["abs_ic"], reverse=True)

    # ── Category breakdowns (raw IC) ──────────────────────────────────────────
    def _top_in_category(col_prefixes_or_list, n=10):
        def _match(col):
            return any(col.startswith(p) or col == p
                       for p in col_prefixes_or_list)
        return sorted(
            [r for r in results if _match(r["col"])],
            key=lambda r: r["max_abs_ic"], reverse=True
        )[:n]

    trend_prefixes     = ["MACD", "ADX", "AROON", "EMA", "SMA", "WMA", "SUPERT",
                          "PSAR", "TRIX", "KST", "DPO", "VI_", "ICH", "close_vs_EMA"]
    momentum_prefixes  = ["RSI", "STOCH", "WILLR", "ROC", "AO", "CMO", "FISHER",
                          "UO", "PPO", "CCI", "MOM", "KDJ", "STC", "COPPOCK"]
    volume_prefixes    = ["MFI", "CMF", "volume", "OBV", "AD_", "VWAP"]
    volatility_prefixes= ["BB_", "KC_", "DC_", "ATR", "NATR", "HV_", "UI_"]
    pattern_prefixes   = ["CDL_"]

    htf_top = {}
    for htf in higher_tfs:
        prefix = f"HTF_{htf}_"
        htf_r  = sorted(
            [r for r in results if r["col"].startswith(prefix)],
            key=lambda r: r["max_abs_ic"], reverse=True
        )[:15]
        htf_top[htf] = htf_r

    # Top IC per regime
    top_trending = sorted(
        [r for r in results if r.get("regime_ic", {}).get("trending_up")],
        key=lambda r: abs(r["regime_ic"].get("trending_up", 0)), reverse=True
    )[:15]
    top_ranging = sorted(
        [r for r in results if r.get("regime_ic", {}).get("ranging")],
        key=lambda r: abs(r["regime_ic"].get("ranging", 0)), reverse=True
    )[:15]

    final_result = {
        "pairs_analyzed":      pairs_done,
        "indicators_analyzed": len(results),
        "primary_tf":          primary_tf,
        "higher_tfs":          higher_tfs,
        # Raw indicator IC
        "top_overall":         [r for r in results if r["max_abs_ic"] >= 0.03][:100],
        "top_trend":           _top_in_category(trend_prefixes),
        "top_momentum":        _top_in_category(momentum_prefixes),
        "top_volume":          _top_in_category(volume_prefixes),
        "top_volatility":      _top_in_category(volatility_prefixes),
        "top_patterns":        _top_in_category(pattern_prefixes),
        "top_trending_regime": top_trending,
        "top_ranging_regime":  top_ranging,
        "htf_top":             htf_top,
        "forward_periods":     FORWARD_PERIODS,
        # Condition IC
        "top_conditions":      cond_results[:50],
        "n_conditions_tested": len(cond_results),
        # Combination IC
        "top_combos":          combo_results[:30],
        "top_combos_synergistic": [r for r in combo_results if r["synergistic"]][:15],
        "n_combos_tested":     len(combo_results),
    }

    msg = (
        f"IC complete: {len(results)} indicators, {len(cond_results)} conditions, "
        f"{len(combo_results)} combinations on {pairs_done} pairs. "
        f"Top: {results[0]['col']} IC={results[0]['best_ic']:.4f}"
        if results else "No results."
    )
    m.update_task(db_path, task_id,
                  status="done",
                  progress=len(sample),
                  total=len(sample),
                  message=msg,
                  result=final_result)
