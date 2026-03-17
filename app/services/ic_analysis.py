"""
Information Coefficient (IC) Analysis Service.

The IC is the rank correlation between an indicator value at bar N
and the forward return N bars later. This is the core question in
quantitative research: "Does this indicator actually PREDICT future returns?"

IC values:
  |IC| > 0.05  — weak but meaningful (worth testing)
  |IC| > 0.10  — moderate predictive power
  |IC| > 0.20  — strong (rare in practice)
  |IC| > 0.30  — very strong (be careful, check for lookahead)

ICIR (IC Information Ratio) = mean(IC) / std(IC)
  ICIR > 0.5  — consistent signal
  ICIR > 1.0  — strong, reliable signal
  ICIR > 2.0  — exceptional

Market Regime Analysis:
  Additionally segments data into trending vs ranging and high vs low volatility
  to determine which regime each indicator works best in.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .. import models as m
from ..strategies.base import BaseStrategy, inject_htf_features, HTF_INJECT_COLS
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)

FORWARD_PERIODS = [1, 5, 10, 20, 40]  # bars ahead

# Indicators to analyze — all numeric columns added by populate_indicators
ANALYSIS_COLS = [
    # Trend
    "MACD_hist", "ADX_14", "DMP_14", "DMN_14", "AROON_osc",
    "EMA50_slope", "close_vs_EMA_20", "close_vs_EMA_50", "close_vs_EMA_200",
    "ema_aligned_bull", "PSAR_dir", "SUPERT_dir",
    "ICH_above_cloud", "ICH_below_cloud",
    "TRIX_15", "KST", "DPO_20", "VI_pos", "VI_neg",
    # Momentum
    "RSI_14", "RSI_7", "RSI_21", "STOCH_K", "STOCH_D",
    "STOCHRSI_K", "WILLR_14", "ROC_10", "ROC_20",
    "AO", "CMO_14", "FISHER", "UO", "PPO", "PPO_hist", "CCI_20",
    # Volatility
    "BB_pct_20", "BB_width_20", "ATR_14", "NATR_14", "HV_20",
    "BB_SQUEEZE", "HIGH_VOL_REGIME",
    # Volume
    "MFI_14", "CMF_20", "volume_ratio", "OBV_trend",
    # Price action
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "PRICE_RANGE_PCT", "close_pct_change",
    "PP_dist_pct", "BARS_SINCE_HIGH", "BARS_SINCE_LOW",
    # Candlestick patterns
    "CDL_BULL_ENGULFING", "CDL_BEAR_ENGULFING",
    "CDL_HAMMER", "CDL_SHOOTING_STAR",
    "CDL_MORNING_STAR", "CDL_EVENING_STAR",
    "CDL_3_WHITE_SOLDIERS", "CDL_3_BLACK_CROWS",
    "CDL_BULL_MARUBOZU", "CDL_BEAR_MARUBOZU",
    "CDL_DOJI", "CDL_DRAGONFLY", "CDL_GRAVESTONE",
    "CDL_DARK_CLOUD", "CDL_PIERCING",
]


def _ic_for_indicator(series: pd.Series, forward_returns: dict[int, pd.Series],
                      min_obs: int = 30) -> dict:
    """
    Compute IC (Spearman rank correlation) between an indicator and
    forward returns at multiple horizons.

    Returns dict: {period: ic_value, ...} plus stability metrics.
    """
    result = {}
    ic_values_by_period = {}

    for period, fwd_ret in forward_returns.items():
        # Align and drop NaN
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

    # Rolling IC stability — vectorized with pandas rolling rank correlation.
    # Pearson correlation on ranks == Spearman; replaces ~20 scipy.spearmanr()
    # calls per indicator with a single vectorized rolling operation.
    primary_period = 10
    if primary_period in forward_returns:
        fwd = forward_returns[primary_period]
        aligned = pd.concat([series, fwd], axis=1).dropna()
        if len(aligned) >= 60:
            window = max(30, len(aligned) // 5)
            x_rank = aligned.iloc[:, 0].rank()
            y_rank = aligned.iloc[:, 1].rank()
            with np.errstate(invalid="ignore", divide="ignore"):
                rolling_corr = x_rank.rolling(window=window, min_periods=max(20, window // 2)).corr(y_rank)
            step = max(1, len(aligned) // 20)
            roll_ics = [v for v in rolling_corr.iloc[::step].tolist() if not np.isnan(v)]
            if len(roll_ics) >= 3:
                ic_mean = float(np.mean(roll_ics))
                ic_std  = float(np.std(roll_ics))
                result["ic_mean"]   = round(ic_mean, 4)
                result["ic_std"]    = round(ic_std, 4)
                result["icir"]      = round(ic_mean / ic_std, 3) if ic_std > 1e-6 else 0.0
                result["ic_positive_pct"] = round(sum(1 for v in roll_ics if v > 0) / len(roll_ics), 3)

    # Best and worst horizon
    valid_ics = {k: v for k, v in ic_values_by_period.items() if not np.isnan(v)}
    if valid_ics:
        result["best_period"]    = max(valid_ics, key=lambda k: abs(valid_ics[k]))
        result["best_ic"]        = round(valid_ics[result["best_period"]], 4)
        result["max_abs_ic"]     = round(abs(result["best_ic"]), 4)
        result["direction"]      = "bullish" if result["best_ic"] > 0 else "bearish"

    return result


def _regime_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split DataFrame into market regime buckets:
    - trending_up: EMA50 slope > 0.05%
    - trending_down: EMA50 slope < -0.05%
    - ranging: |EMA50 slope| <= 0.05%
    - high_vol: normalized ATR in top quartile
    - low_vol: normalized ATR in bottom quartile
    """
    regimes = {}

    if "EMA50_slope" in df.columns:
        slope = df["EMA50_slope"]
        regimes["trending_up"]   = df[slope > 0.05]
        regimes["trending_down"] = df[slope < -0.05]
        regimes["ranging"]       = df[slope.abs() <= 0.05]

    if "NATR_14" in df.columns:
        natr = df["NATR_14"].dropna()
        q25 = natr.quantile(0.25)
        q75 = natr.quantile(0.75)
        regimes["high_vol"] = df[df["NATR_14"] >= q75]
        regimes["low_vol"]  = df[df["NATR_14"] <= q25]

    return regimes


def _build_htf_analysis_cols(timeframes: list) -> list:
    """Build the list of HTF indicator column names to include in IC analysis."""
    htf_cols = []
    primary_tf = timeframes[0] if timeframes else "1h"
    for tf in timeframes[1:]:
        for base_col in HTF_INJECT_COLS:
            htf_cols.append(f"HTF_{tf}_{base_col}")
        htf_cols.append(f"HTF_{tf}_trend_dir")
        htf_cols.append(f"HTF_{tf}_regime")
    return htf_cols


def run_ic_analysis(task_id: str, db_path: Path, session_id: str,
                    parquet_dir: Path, timeframes: list,stop_event=None) -> None:
    """
    Background task: compute IC for every indicator across all pairs and all timeframes.

    Multi-timeframe approach:
    - Primary TF indicators are computed on the primary (lowest) timeframe.
    - Higher TF indicators are injected as HTF_{tf}_ prefixed columns using
      forward-fill so each primary bar knows its higher-TF context.
    - IC is computed for both primary and HTF indicators against the primary TF
      forward returns (the actual question: does HTF RSI predict LTF returns?).
    - Results are aggregated across pairs (mean IC = more robust estimate).
    """
    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    pairs = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    primary_tf = timeframes[0] if timeframes else "1h"
    higher_tfs = timeframes[1:] if len(timeframes) > 1 else []

    if not active:
        m.update_task(db_path, task_id, status="error",
                      error="No active pairs with data.",
                      message="Download data and add indicators first.")
        return

    # Build full analysis column list including HTF columns
    htf_analysis_cols = _build_htf_analysis_cols(timeframes)
    all_analysis_cols = ANALYSIS_COLS + htf_analysis_cols

    # Sample up to 30 pairs for performance
    sample = active[:30]
    tf_desc = f"{primary_tf}" + (f" + HTF: {', '.join(higher_tfs)}" if higher_tfs else "")
    progress(0, len(sample), f"Computing IC for {len(sample)} pairs [{tf_desc}]…")

    base_strategy = BaseStrategy()
    # Accumulate IC values per indicator, per forward period
    # { indicator: { period: [ic_values across pairs] } }
    pair_ics: dict[str, dict] = {}
    regime_ics: dict[str, dict[str, dict]] = {}  # {indicator: {regime: [ic_values]}}

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

        # Populate indicators if not already present
        if "RSI_14" not in df.columns:
            df = base_strategy.populate_indicators(df)
        if "SUPERT_dir" not in df.columns or df["SUPERT_dir"].isna().all():
            df = base_strategy.populate_indicators(df)

        # ── Inject HTF features ───────────────────────────────────────────────
        for htf in higher_tfs:
            htf_path = parquet_path(parquet_dir, session_id, pair["symbol"], htf)
            if not htf_path.exists():
                continue
            try:
                htf_df = pd.read_parquet(htf_path)
                df = inject_htf_features(df, htf_df, htf, base_strategy)
            except Exception as e:
                logger.warning("IC: HTF inject failed %s %s: %s", pair["symbol"], htf, e)

        # Compute forward returns for each period
        fwd_returns: dict[int, pd.Series] = {}
        for period in FORWARD_PERIODS:
            fwd_returns[period] = df["close"].pct_change(period).shift(-period)

        # Compute IC for each indicator (primary + all HTF columns)
        for col in all_analysis_cols:
            if col not in df.columns:
                continue
            series = df[col]
            if series.isna().all():
                continue

            ic_data = _ic_for_indicator(series, fwd_returns)

            if col not in pair_ics:
                pair_ics[col] = {f"ic_{p}": [] for p in FORWARD_PERIODS}
                pair_ics[col]["icir_list"] = []
                pair_ics[col]["ic_positive_pct_list"] = []

            for period in FORWARD_PERIODS:
                val = ic_data.get(f"ic_{period}")
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    pair_ics[col][f"ic_{period}"].append(val)

            if "icir" in ic_data and not np.isnan(ic_data["icir"]):
                pair_ics[col]["icir_list"].append(ic_data["icir"])
            if "ic_positive_pct" in ic_data:
                pair_ics[col]["ic_positive_pct_list"].append(ic_data["ic_positive_pct"])

            # Regime-conditional IC
            regimes = _regime_split(df)
            for regime_name, regime_df in regimes.items():
                if len(regime_df) < 50:
                    continue
                if col not in regime_df.columns:
                    continue
                r_series = regime_df[col]
                r_fwd = {p: regime_df["close"].pct_change(p).shift(-p) for p in [10]}
                r_ic = _ic_for_indicator(r_series, r_fwd, min_obs=20)
                r_ic_val = r_ic.get("ic_10")
                if r_ic_val is not None and not (isinstance(r_ic_val, float) and np.isnan(r_ic_val)):
                    if col not in regime_ics:
                        regime_ics[col] = {}
                    if regime_name not in regime_ics[col]:
                        regime_ics[col][regime_name] = []
                    regime_ics[col][regime_name].append(r_ic_val)

        pairs_done += 1

    if not pair_ics:
        m.update_task(db_path, task_id, status="error",
                      error="No indicators found in data. Run Add Indicators first.",
                      message="Run Phase 5 (Add Indicators) before IC analysis.")
        return

    progress(len(sample), len(sample), "Aggregating results…")

    # Aggregate across pairs: mean IC per period
    results = []
    for col, period_data in pair_ics.items():
        row = {"col": col}
        valid_ics = {}

        for period in FORWARD_PERIODS:
            vals = period_data.get(f"ic_{period}", [])
            if vals:
                avg_ic = float(np.mean(vals))
                row[f"ic_{period}"] = round(avg_ic, 4)
                valid_ics[period] = avg_ic
            else:
                row[f"ic_{period}"] = None

        if not valid_ics:
            continue

        # ICIR across pairs
        icir_vals = period_data.get("icir_list", [])
        row["icir"] = round(float(np.mean(icir_vals)), 3) if icir_vals else None

        # IC positive percentage (how often IC is positive = consistent direction)
        pos_pct_vals = period_data.get("ic_positive_pct_list", [])
        row["ic_positive_pct"] = round(float(np.mean(pos_pct_vals)), 3) if pos_pct_vals else None

        # Best horizon
        abs_ics = {p: abs(v) for p, v in valid_ics.items()}
        best_p = max(abs_ics, key=abs_ics.get)
        row["best_period"] = best_p
        row["best_ic"]     = round(valid_ics[best_p], 4)
        row["max_abs_ic"]  = round(abs(valid_ics[best_p]), 4)
        row["direction"]   = "bullish" if valid_ics[best_p] > 0 else "bearish"

        # Regime IC
        if col in regime_ics:
            row["regime_ic"] = {
                regime: round(float(np.mean(vals)), 4)
                for regime, vals in regime_ics[col].items()
                if vals
            }
            # Best regime
            if row["regime_ic"]:
                best_regime = max(row["regime_ic"], key=lambda k: abs(row["regime_ic"][k]))
                row["best_regime"] = best_regime
                row["best_regime_ic"] = row["regime_ic"][best_regime]
        else:
            row["regime_ic"] = {}

        # Significance label
        max_abs = row["max_abs_ic"]
        if max_abs >= 0.20:
            row["strength"] = "strong"
        elif max_abs >= 0.10:
            row["strength"] = "moderate"
        elif max_abs >= 0.05:
            row["strength"] = "weak"
        else:
            row["strength"] = "noise"

        results.append(row)

    # Sort by best absolute IC descending
    results.sort(key=lambda r: r["max_abs_ic"], reverse=True)

    # Top results for each category
    top_overall    = [r for r in results if r["max_abs_ic"] >= 0.03][:30]
    top_trending   = sorted(
        [r for r in results if r.get("regime_ic", {}).get("trending_up")],
        key=lambda r: abs(r["regime_ic"].get("trending_up", 0)), reverse=True
    )[:10]
    top_ranging    = sorted(
        [r for r in results if r.get("regime_ic", {}).get("ranging")],
        key=lambda r: abs(r["regime_ic"].get("ranging", 0)), reverse=True
    )[:10]

    # Indicator category breakdown
    trend_cols = ["MACD_hist","ADX_14","AROON_osc","EMA50_slope","SUPERT_dir","PSAR_dir","TRIX_15","CCI_20"]
    momentum_cols = ["RSI_14","RSI_7","STOCH_K","STOCHRSI_K","WILLR_14","CMO_14","FISHER","AO","UO","PPO_hist"]
    volume_cols = ["MFI_14","CMF_20","volume_ratio","OBV_trend"]
    volatility_cols = ["BB_pct_20","BB_width_20","NATR_14","HV_20","BB_SQUEEZE","ATR_14"]
    pattern_cols = [c for c in ANALYSIS_COLS if c.startswith("CDL_")]

    # HTF breakdowns: one entry per higher TF
    htf_top = {}
    for htf in higher_tfs:
        htf_col_prefix = f"HTF_{htf}_"
        htf_results = [r for r in results if r["col"].startswith(htf_col_prefix)]
        htf_results.sort(key=lambda r: r["max_abs_ic"], reverse=True)
        htf_top[htf] = htf_results[:10]

    def top_in_category(category_cols):
        return sorted(
            [r for r in results if r["col"] in category_cols],
            key=lambda r: r["max_abs_ic"], reverse=True
        )[:5]

    final_result = {
        "pairs_analyzed": pairs_done,
        "indicators_analyzed": len(results),
        "primary_tf": primary_tf,
        "higher_tfs": higher_tfs,
        "top_overall": top_overall,
        "top_trend": top_in_category(trend_cols),
        "top_momentum": top_in_category(momentum_cols),
        "top_volume": top_in_category(volume_cols),
        "top_volatility": top_in_category(volatility_cols),
        "top_patterns": top_in_category(pattern_cols),
        "top_trending_regime": top_trending,
        "top_ranging_regime": top_ranging,
        "htf_top": htf_top,
        "forward_periods": FORWARD_PERIODS,
    }

    msg = (f"IC analysis complete: {len(results)} indicators analyzed on {pairs_done} pairs. "
           f"Top IC: {results[0]['col']} = {results[0]['best_ic']:.4f} "
           f"at {results[0]['best_period']}-bar horizon." if results else "No results.")

    m.update_task(db_path, task_id,
                  status="done",
                  progress=len(sample),
                  total=len(sample),
                  message=msg,
                  result=final_result)
