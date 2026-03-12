"""
Phase 8: Algo Finder service.
Uses Optuna to search for the best combination of indicator rules
that maximises profitability.

Two modes
─────────
  Path B (default): Free search across a user-selected set of indicators.
  Path A: User-defined base entry logic is fixed; Optuna only tunes
          filter thresholds identified by the winner/loser analysis.

Indicator search design (Path B)
─────────────────────────────────
  Each indicator in INDICATOR_CATALOG has a FIXED, semantically correct
  condition type.  Optuna never mixes incompatible units (e.g. ATR vs close).
  Instead it searches:
    • use_<key>   — whether to include this indicator at all (0/1 toggle)
    • thresh_<key>— the threshold value (for lt / gt condition types)

  Condition types
  ───────────────
    lt          col < threshold          (oscillators in oversold territory)
    gt          col > threshold          (trend strength, volume ratio, ROC)
    gt_zero     col > 0                  (MACD hist, AO, CMF, OBV trend sign)
    eq1         col == 1                 (Supertrend, PSAR: direction flags)
    price_gt    close > col              (price above a moving average)
    price_lt    close < col              (price below a band — oversold)
    ema_cross   EMA_fast > EMA_slow      (parameterised EMA alignment)
    col_gt_col  col[0] > col[1]          (Aroon up > down)
"""
import logging
import os
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy, inject_htf_features
from ..services.backtest import _simple_backtest, _walk_forward_backtest
from ..services.entry_logic_analyzer import _eval_entry_condition, _parse_direction
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Indicator catalogue ───────────────────────────────────────────────────────
# Every entry maps a short key → condition spec.
# "default": True  →  pre-selected when the user hasn't customised anything.

INDICATOR_CATALOG = OrderedDict([
    # ── Oscillators: oversold entry conditions ──────────────────────────
    ("rsi_14",    {"col": "RSI_14",      "type": "lt",  "range": (20, 50),
                   "label": "RSI(14) oversold",         "cat": "Oscillators", "default": True}),
    ("rsi_7",     {"col": "RSI_7",       "type": "lt",  "range": (15, 45),
                   "label": "RSI(7) oversold",          "cat": "Oscillators"}),
    ("rsi_21",    {"col": "RSI_21",      "type": "lt",  "range": (25, 55),
                   "label": "RSI(21) oversold",         "cat": "Oscillators"}),
    ("stoch_k",   {"col": "STOCH_K",     "type": "lt",  "range": (10, 40),
                   "label": "Stoch %K oversold",        "cat": "Oscillators"}),
    ("stochrsi",  {"col": "STOCHRSI_K",  "type": "lt",  "range": (5, 30),
                   "label": "StochRSI %K oversold",     "cat": "Oscillators"}),
    ("willr",     {"col": "WILLR_14",    "type": "lt",  "range": (-80, -20),
                   "label": "Williams %R oversold",     "cat": "Oscillators"}),
    ("mfi_os",    {"col": "MFI_14",      "type": "lt",  "range": (20, 50),
                   "label": "MFI(14) oversold",         "cat": "Oscillators"}),
    ("cci",       {"col": "CCI_20",      "type": "lt",  "range": (-150, -50),
                   "label": "CCI(20) oversold",         "cat": "Oscillators"}),

    # ── Trend: direction and alignment ──────────────────────────────────
    ("ema_cross",  {"col": ("EMA_fast", "EMA_slow"), "type": "ema_cross",
                    "label": "EMA fast > EMA slow",     "cat": "Trend", "default": True}),
    ("close_ema50", {"col": "EMA_50",   "type": "price_gt",
                     "label": "Price > EMA(50)",        "cat": "Trend"}),
    ("close_ema200",{"col": "EMA_200",  "type": "price_gt",
                     "label": "Price > EMA(200)",       "cat": "Trend"}),
    ("supertrend",  {"col": "SUPERT_dir","type": "eq1",
                     "label": "Supertrend Bullish",     "cat": "Trend", "default": True}),
    ("psar",        {"col": "PSAR_dir",  "type": "eq1",
                     "label": "Parabolic SAR Bullish",  "cat": "Trend"}),
    ("aroon_bull",  {"col": ("AROON_up", "AROON_down"), "type": "col_gt_col",
                     "label": "Aroon Up > Aroon Down",  "cat": "Trend"}),
    ("adx_min",     {"col": "ADX_14",    "type": "gt",  "range": (15, 35),
                     "label": "ADX(14) trend strength", "cat": "Trend", "default": True}),

    # ── Momentum: directional strength ──────────────────────────────────
    ("macd_hist",  {"col": "MACD_hist",  "type": "gt_zero",
                    "label": "MACD Histogram > 0",      "cat": "Momentum", "default": True}),
    ("roc",        {"col": "ROC_10",     "type": "gt",  "range": (-1.0, 3.0),
                    "label": "ROC(10) > threshold %",   "cat": "Momentum"}),
    ("ao",         {"col": "AO",         "type": "gt_zero",
                    "label": "Awesome Oscillator > 0",  "cat": "Momentum"}),

    # ── Volatility & Bands ───────────────────────────────────────────────
    ("bb_pct",    {"col": "BB_pct_20",   "type": "lt",  "range": (0.1, 0.4),
                   "label": "BB %B near lower band",    "cat": "Volatility", "default": True}),
    ("bb_width",  {"col": "BB_width_20", "type": "gt",  "range": (0.005, 0.05),
                   "label": "BB Width > min (not squeezed)", "cat": "Volatility"}),
    ("kelt_lower",{"col": "KC_lower",    "type": "price_lt",
                   "label": "Price < Keltner Lower (oversold)", "cat": "Volatility"}),
    ("natr",      {"col": "NATR_14",     "type": "lt",  "range": (0.5, 4.0),
                   "label": "NATR(14) < max % (low vol entry)", "cat": "Volatility"}),

    # ── Volume: participation filters ────────────────────────────────────
    ("vol_ratio", {"col": "volume_ratio","type": "gt",  "range": (0.5, 2.5),
                   "label": "Volume Ratio above average","cat": "Volume", "default": True}),
    ("cmf",       {"col": "CMF_20",      "type": "gt_zero",
                   "label": "Chaikin MF > 0 (buying pressure)", "cat": "Volume"}),
    ("obv_trend", {"col": "OBV_trend",   "type": "eq1",
                   "label": "OBV Trend Bullish",        "cat": "Volume"}),
    ("mfi_bull",  {"col": "MFI_14",      "type": "gt",  "range": (40, 65),
                   "label": "MFI(14) > threshold (money flowing in)", "cat": "Volume"}),
])

DEFAULT_INDICATORS = [k for k, v in INDICATOR_CATALOG.items() if v.get("default")]

# HTF filters: always available in addition to whatever primary TF indicators are chosen
_HTF_SEARCH = {
    "use_htf_trend":      (0, 1),
    "use_htf_supertrend": (0, 1),
    "htf_rsi_max":        (30, 70),
    "use_htf_rsi_filter": (0, 1),
    "htf_adx_min":        (15, 40),
    "use_htf_adx_filter": (0, 1),
}


# ── Strategy class ────────────────────────────────────────────────────────────

class CatalogStrategy(BaseStrategy):
    """
    Strategy built dynamically from a user-selected subset of INDICATOR_CATALOG.
    Optuna controls:
      • use_<key>   — whether the indicator is active this trial
      • thresh_<key>— the threshold value (for lt / gt types)
      • ema_fast / ema_slow — EMA periods (only when ema_cross is selected)
    """
    name = "CatalogStrategy"

    def __init__(self, params: dict, selected: list):
        super().__init__(params)
        self._selected = selected

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        cond = pd.Series(True, index=df.index)

        for key in self._selected:
            if not p.get(f"use_{key}", 0):
                continue
            spec = INDICATOR_CATALOG.get(key)
            if spec is None:
                continue

            col  = spec["col"]
            kind = spec["type"]

            if kind == "lt":
                if col in df.columns:
                    cond &= df[col] < p[f"thresh_{key}"]

            elif kind == "gt":
                if col in df.columns:
                    cond &= df[col] > p[f"thresh_{key}"]

            elif kind == "gt_zero":
                if col in df.columns:
                    cond &= df[col] > 0

            elif kind == "eq1":
                if col in df.columns:
                    cond &= df[col] == 1

            elif kind == "price_gt":
                # close > moving average column
                if col in df.columns:
                    cond &= df["close"] > df[col]

            elif kind == "price_lt":
                # close < band column (oversold below lower band)
                if col in df.columns:
                    cond &= df["close"] < df[col]

            elif kind == "ema_cross":
                # EMA_fast > EMA_slow (both periods are Optuna params)
                fc = f"EMA_{int(p.get('ema_fast', 20))}"
                sc = f"EMA_{int(p.get('ema_slow', 50))}"
                if fc in df.columns and sc in df.columns:
                    cond &= df[fc] > df[sc]

            elif kind == "col_gt_col":
                c1, c2 = col
                if c1 in df.columns and c2 in df.columns:
                    cond &= df[c1] > df[c2]

        # ── HTF filters ──────────────────────────────────────────────────
        htf_trend_cols = [c for c in df.columns if c.endswith("_trend_dir")]
        if p.get("use_htf_trend", 0) and htf_trend_cols:
            cond &= df[htf_trend_cols[0]] == 1

        htf_supert_cols = [c for c in df.columns if "HTF_" in c and c.endswith("_SUPERT_dir")]
        if p.get("use_htf_supertrend", 0) and htf_supert_cols:
            cond &= df[htf_supert_cols[0]] == 1

        htf_rsi_cols = [c for c in df.columns if "HTF_" in c and c.endswith("_RSI_14")]
        if p.get("use_htf_rsi_filter", 0) and htf_rsi_cols:
            cond &= df[htf_rsi_cols[0]] < p.get("htf_rsi_max", 60)

        htf_adx_cols = [c for c in df.columns if "HTF_" in c and c.endswith("_ADX_14")]
        if p.get("use_htf_adx_filter", 0) and htf_adx_cols:
            cond &= df[htf_adx_cols[0]] > p.get("htf_adx_min", 20)

        df["entry_signal"] = cond.astype(int)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rsi_col = f"RSI_{int(p.get('rsi_exit_period', 14))}"
        cond = pd.Series(False, index=df.index)
        if rsi_col in df.columns:
            cond |= df[rsi_col] > p.get("rsi_exit_min", 70)
        if "ema_20_50_cross" in df.columns:
            cond |= df["ema_20_50_cross"] == -1
        df["exit_signal"] = cond.astype(int)
        return df


# ── Optuna objective ──────────────────────────────────────────────────────────

def _objective(trial, dfs: list, config: dict,
               selected: list, has_htf: bool = False) -> float:
    """Optuna objective: build params from catalog selection, return OOS Sharpe."""
    params = {}

    # EMA cross needs fast/slow period parameters
    if "ema_cross" in selected:
        params["ema_fast"] = trial.suggest_int("ema_fast", 8, 30)
        params["ema_slow"] = trial.suggest_int("ema_slow", 20, 100)
        if params["ema_fast"] >= params["ema_slow"]:
            return -999.0

    # Exit parameters (always tuned)
    params["rsi_exit_period"] = trial.suggest_categorical("rsi_exit_period", [7, 14, 21])
    params["rsi_exit_min"]    = trial.suggest_float("rsi_exit_min", 60, 85)

    # Per-indicator: on/off toggle + threshold (when applicable)
    for key in selected:
        spec = INDICATOR_CATALOG.get(key)
        if spec is None:
            continue
        params[f"use_{key}"] = trial.suggest_categorical(f"use_{key}", [0, 1])
        if spec["type"] in ("lt", "gt") and "range" in spec:
            lo, hi = spec["range"]
            params[f"thresh_{key}"] = trial.suggest_float(f"thresh_{key}", lo, hi)

    # HTF filters (only when HTF data present)
    if has_htf:
        params["use_htf_trend"]      = trial.suggest_categorical("use_htf_trend", [0, 1])
        params["use_htf_supertrend"] = trial.suggest_categorical("use_htf_supertrend", [0, 1])
        params["use_htf_rsi_filter"] = trial.suggest_categorical("use_htf_rsi_filter", [0, 1])
        params["htf_rsi_max"]        = trial.suggest_float("htf_rsi_max", 30, 70)
        params["use_htf_adx_filter"] = trial.suggest_categorical("use_htf_adx_filter", [0, 1])
        params["htf_adx_min"]        = trial.suggest_float("htf_adx_min", 15, 40)

    strategy = CatalogStrategy(params, selected)
    all_wfo = []

    for df in dfs:
        if len(df) < 100:
            continue
        enriched = df.copy()
        enriched = strategy.populate_entry_signal(enriched)
        enriched = strategy.populate_exit_signal(enriched)
        wfo = _walk_forward_backtest(
            enriched, strategy,
            n_splits=config.get("wfo_splits", 3),
            train_ratio=config.get("wfo_train_ratio", 0.7),
            initial_capital=config.get("initial_capital", 10_000),
            fee_rate=config.get("fee_rate", 0.001),
            slippage=config.get("slippage", 0.0005),
            position_size=config.get("position_size", 0.1),
            fast_mode=True,
        )
        all_wfo.append(wfo)

    if not all_wfo:
        return -999.0

    oos_sharpe = np.mean([w["oos_sharpe"] for w in all_wfo])
    oos_trades = np.mean([w["oos_trades"] for w in all_wfo])
    oos_return = np.mean([w["oos_return"] for w in all_wfo])

    if oos_trades < 5:
        return -999.0

    return float(oos_sharpe + oos_return * 0.01)


# ── Main task ─────────────────────────────────────────────────────────────────

def run_algofinder(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list,
                   n_trials: int = 50, config: dict = None) -> None:
    """
    Background task: run Optuna search for best strategy parameters.
    config["selected_indicators"] controls which indicators are searched.
    Falls back to DEFAULT_INDICATORS when not specified.
    """
    if config is None:
        config = {}

    selected = config.get("selected_indicators") or DEFAULT_INDICATORS
    # Keep only keys that exist in the catalog
    selected = [k for k in selected if k in INDICATOR_CATALOG]
    if not selected:
        selected = DEFAULT_INDICATORS

    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    progress(0, n_trials, "Loading data for Algo Finder…")

    pairs  = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    primary_tf = timeframes[0] if timeframes else "1h"
    higher_tfs = timeframes[1:] if len(timeframes) > 1 else []

    dfs = []
    htf_found_flag = [False]
    load_lock = threading.Lock()

    def _load_pair(pair: dict):
        symbol = pair["symbol"]
        path = parquet_path(parquet_dir, session_id, symbol, primary_tf)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if len(df) < 100:
            return None
        strat = BaseStrategy()
        df = strat.populate_indicators(df)
        pair_htf_found = False
        for htf in higher_tfs:
            htf_path = parquet_path(parquet_dir, session_id, symbol, htf)
            if not htf_path.exists():
                continue
            try:
                htf_df = pd.read_parquet(htf_path)
                df = inject_htf_features(df, htf_df, htf, strat)
                pair_htf_found = True
            except Exception as e:
                logger.warning("AlgoFinder HTF inject %s %s: %s", symbol, htf, e)
        if pair_htf_found:
            with load_lock:
                htf_found_flag[0] = True
        return df

    sample_pairs = active[:20]
    workers = min(os.cpu_count() or 4, len(sample_pairs), 4)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        for result in exe.map(_load_pair, sample_pairs):
            if result is not None:
                dfs.append(result)

    htf_found = htf_found_flag[0]

    if not dfs:
        m.update_task(db_path, task_id, status="error",
                      error="No data available for Algo Finder.",
                      message="Please download and enrich data first.")
        return

    sel_labels = ", ".join(
        INDICATOR_CATALOG[k]["label"] for k in selected if k in INDICATOR_CATALOG
    )
    tf_desc = primary_tf + (f" + HTF: {', '.join(higher_tfs)}" if htf_found else "")
    progress(0, n_trials,
             f"Running {n_trials} trials on {len(dfs)} pairs [{tf_desc}] — "
             f"{len(selected)} indicator groups…")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    trial_count = [0]
    counter_lock = threading.Lock()

    def callback(study, trial):
        with counter_lock:
            trial_count[0] += 1
            count = trial_count[0]
        if count % 5 == 0:
            try:
                best = study.best_value
            except Exception:
                best = float("nan")
            progress(count, n_trials, f"Trial {count}/{n_trials} — best score: {best:.3f}")

    n_jobs = min(os.cpu_count() or 1, 4)
    study.optimize(
        lambda trial: _objective(trial, dfs, config, selected, has_htf=htf_found),
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[callback],
        show_progress_bar=False,
    )

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value > -100]
    completed.sort(key=lambda t: t.value, reverse=True)
    top5 = completed[:5]

    for rank, trial in enumerate(top5, 1):
        params = trial.params
        strategy = CatalogStrategy(params, selected)

        all_wfo = []
        for df in dfs:
            enriched = df.copy()
            enriched = strategy.populate_entry_signal(enriched)
            enriched = strategy.populate_exit_signal(enriched)
            wfo = _walk_forward_backtest(
                enriched, strategy,
                n_splits=config.get("wfo_splits", 3),
                train_ratio=config.get("wfo_train_ratio", 0.7),
                initial_capital=config.get("initial_capital", 10_000),
                fee_rate=config.get("fee_rate", 0.001),
                slippage=config.get("slippage", 0.0005),
                position_size=config.get("position_size", 0.1),
            )
            all_wfo.append(wfo)

        def avg(key):
            vals = [w[key] for w in all_wfo]
            return float(np.mean(vals)) if vals else 0.0

        rules = _describe_rules(params, selected)
        m.save_algo_result(
            db_path, session_id, None, rank,
            f"AlgoStrategy_#{rank}",
            params, rules,
            {
                "is_return":    avg("is_return"),
                "oos_return":   avg("oos_return"),
                "win_rate":     avg("oos_win_rate"),
                "sharpe":       avg("oos_sharpe"),
                "max_drawdown": avg("oos_max_dd"),
            },
        )

    progress(n_trials, n_trials,
             f"Algo Finder complete — {len(top5)} strategies found.")
    m.update_task(db_path, task_id, result={"top_count": len(top5)})


# ── Rule description ──────────────────────────────────────────────────────────

def _describe_rules(params: dict, selected: list) -> str:
    """Convert params dict to a human-readable rule description."""
    _COND_TMPL = {
        "lt":        lambda spec, p, k: f"{spec['col']} < {p[f'thresh_{k}']:.4g}",
        "gt":        lambda spec, p, k: f"{spec['col']} > {p[f'thresh_{k}']:.4g}",
        "gt_zero":   lambda spec, p, k: f"{spec['col']} > 0",
        "eq1":       lambda spec, p, k: f"{spec['col']} = Bullish (1)",
        "price_gt":  lambda spec, p, k: f"close > {spec['col']}",
        "price_lt":  lambda spec, p, k: f"close < {spec['col']} (oversold)",
        "ema_cross": lambda spec, p, k: (
            f"EMA({p.get('ema_fast', '?')}) > EMA({p.get('ema_slow', '?')})"
        ),
        "col_gt_col":lambda spec, p, k: f"{spec['col'][0]} > {spec['col'][1]}",
    }

    lines = ["ENTRY CONDITIONS:"]
    for key in selected:
        if not params.get(f"use_{key}", 0):
            continue
        spec = INDICATOR_CATALOG.get(key)
        if not spec:
            continue
        tmpl = _COND_TMPL.get(spec["type"])
        if tmpl:
            lines.append(f"  • {tmpl(spec, params, key)}")

    # HTF filters
    if params.get("use_htf_trend"):
        lines.append("  • [HTF] Price > HTF EMA50 (trend aligned)")
    if params.get("use_htf_supertrend"):
        lines.append("  • [HTF] Supertrend Bullish on higher TF")
    if params.get("use_htf_rsi_filter"):
        lines.append(f"  • [HTF] RSI < {params.get('htf_rsi_max', 60):.1f} on higher TF")
    if params.get("use_htf_adx_filter"):
        lines.append(f"  • [HTF] ADX > {params.get('htf_adx_min', 20):.1f} on higher TF")

    lines += [
        "",
        "EXIT CONDITIONS:",
        f"  • RSI({int(params.get('rsi_exit_period', 14))}) > "
        f"{params.get('rsi_exit_min', 70):.1f}",
        "  • EMA(20) crosses below EMA(50)",
    ]
    return "\n".join(lines)


# ── Path A: Fixed entry logic + Optuna filter tuning ─────────────────────────

class PathAStrategy(BaseStrategy):
    """
    Strategy where the base entry signal comes from the user's Python expression
    and Optuna only tunes threshold values for the top discriminative indicator
    filters identified by the winner/loser analysis.
    """
    name = "PathAStrategy"

    def __init__(self, params: dict, condition_code: str, indicator_filters: list):
        super().__init__(params)
        self._condition_code = condition_code
        self._indicator_filters = indicator_filters

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        base_signal = _eval_entry_condition(df, self._condition_code)
        cond = base_signal.copy()
        for f in self._indicator_filters:
            col = f["col"]
            op  = f["operator"]
            threshold = self.params.get(f"threshold_{col}")
            if threshold is None or col not in df.columns:
                continue
            if op == ">":
                cond &= df[col] > threshold
            elif op == "<":
                cond &= df[col] < threshold
        df["entry_signal"] = cond.astype(int)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rsi_period = int(p.get("exit_rsi_period", 14))
        rsi_col    = f"RSI_{rsi_period}"
        rsi_exit   = float(p.get("exit_rsi_threshold", 70))
        cond = pd.Series(False, index=df.index)
        if rsi_col in df.columns:
            cond |= df[rsi_col] > rsi_exit
        if "ema_20_50_cross" in df.columns:
            cond |= df["ema_20_50_cross"] == -1
        df["exit_signal"] = cond.astype(int)
        return df


def _path_a_objective(trial, dfs: list, config: dict,
                      condition_code: str, indicator_filters: list) -> float:
    """Optuna objective for Path A: tune filter thresholds only."""
    params = {}

    for f in indicator_filters:
        col   = f["col"]
        op    = f["operator"]
        w_p25 = f.get("winner_p25", 0.0)
        w_p75 = f.get("winner_p75", 1.0)
        w_mean = f.get("winner_mean", (w_p25 + w_p75) / 2)

        lo = min(w_p25, w_mean) * 0.5
        hi = max(w_p75, w_mean) * 1.5
        if op == "<":
            lo, hi = hi * 0.3, hi * 1.5

        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        if abs(hi - lo) < 1e-8:
            hi = lo + 1.0

        params[f"threshold_{col}"] = trial.suggest_float(f"threshold_{col}", lo, hi)

    params["exit_rsi_period"]    = trial.suggest_categorical("exit_rsi_period", [7, 14, 21])
    params["exit_rsi_threshold"] = trial.suggest_float("exit_rsi_threshold", 60, 85)

    strategy = PathAStrategy(params, condition_code, indicator_filters)
    all_wfo  = []

    for df in dfs:
        if len(df) < 100:
            continue
        enriched = strategy.run(df.copy())
        wfo = _walk_forward_backtest(
            enriched, strategy,
            n_splits=config.get("wfo_splits", 3),
            train_ratio=config.get("wfo_train_ratio", 0.7),
            initial_capital=config.get("initial_capital", 10_000),
            fee_rate=config.get("fee_rate", 0.001),
            slippage=config.get("slippage", 0.0005),
            position_size=config.get("position_size", 0.1),
        )
        all_wfo.append(wfo)

    if not all_wfo:
        return -999.0

    oos_sharpe = np.mean([w["oos_sharpe"] for w in all_wfo])
    oos_trades = np.mean([w["oos_trades"] for w in all_wfo])
    oos_return = np.mean([w["oos_return"] for w in all_wfo])

    if oos_trades < 5:
        return -999.0

    return float(oos_sharpe + oos_return * 0.01)


def run_algofinder_path_a(task_id: str, db_path: Path, session_id: str,
                          parquet_dir: Path, timeframes: list,
                          entry_logic: str, analysis_result: dict,
                          n_trials: int = 50, config: dict = None) -> None:
    """
    Path A Algo Finder: base entry logic is fixed, Optuna tunes filter thresholds.
    """
    if config is None:
        config = {}

    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    direction, condition_code = _parse_direction(entry_logic)
    top_indicators   = analysis_result.get("top_indicators", [])
    indicator_filters = top_indicators[:5]

    if not indicator_filters:
        m.update_task(db_path, task_id, status="error",
                      error="No indicator filters from analysis. Run Entry Logic Analysis first.",
                      message="Go to Entry Logic → Run Analysis first.")
        return

    progress(0, n_trials, f"Loading data for Path A ({len(indicator_filters)} filters)…")

    pairs  = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    primary_tf = timeframes[0] if timeframes else "1h"
    higher_tfs = timeframes[1:] if len(timeframes) > 1 else []

    dfs = []

    def _load_pair_a(pair: dict):
        symbol = pair["symbol"]
        path = parquet_path(parquet_dir, session_id, symbol, primary_tf)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if len(df) < 100:
            return None
        strat = BaseStrategy()
        if "RSI_14" not in df.columns:
            df = strat.populate_indicators(df)
        for htf in higher_tfs:
            htf_path = parquet_path(parquet_dir, session_id, symbol, htf)
            if not htf_path.exists():
                continue
            try:
                htf_df = pd.read_parquet(htf_path)
                df = inject_htf_features(df, htf_df, htf, strat)
            except Exception as e:
                logger.warning("PathA HTF inject %s %s: %s", symbol, htf, e)
        return df

    workers_a = min(os.cpu_count() or 4, len(active[:20]), 4)
    with ThreadPoolExecutor(max_workers=workers_a) as exe:
        for result in exe.map(_load_pair_a, active[:20]):
            if result is not None:
                dfs.append(result)

    if not dfs:
        m.update_task(db_path, task_id, status="error",
                      error="No data available.", message="Download and add indicators first.")
        return

    progress(0, n_trials, f"Running {n_trials} Path A trials on {len(dfs)} pairs…")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    trial_count_a = [0]
    counter_lock_a = threading.Lock()

    def callback(study, trial):
        with counter_lock_a:
            trial_count_a[0] += 1
            count = trial_count_a[0]
        if count % 5 == 0:
            try:
                best = study.best_value
            except Exception:
                best = float("nan")
            progress(count, n_trials, f"Trial {count}/{n_trials} — best: {best:.3f}")

    n_jobs_a = min(os.cpu_count() or 1, 4)
    study.optimize(
        lambda trial: _path_a_objective(
            trial, dfs, config, condition_code, indicator_filters),
        n_trials=n_trials,
        n_jobs=n_jobs_a,
        callbacks=[callback],
        show_progress_bar=False,
    )

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value > -100]
    completed.sort(key=lambda t: t.value, reverse=True)
    top5 = completed[:5]

    for rank, trial in enumerate(top5, 1):
        params   = trial.params
        strategy = PathAStrategy(params, condition_code, indicator_filters)

        all_wfo = []
        for df in dfs:
            enriched = strategy.run(df.copy())
            wfo = _walk_forward_backtest(
                enriched, strategy,
                n_splits=config.get("wfo_splits", 3),
                train_ratio=config.get("wfo_train_ratio", 0.7),
                initial_capital=config.get("initial_capital", 10_000),
                fee_rate=config.get("fee_rate", 0.001),
                slippage=config.get("slippage", 0.0005),
                position_size=config.get("position_size", 0.1),
            )
            all_wfo.append(wfo)

        def avg(key):
            vals = [w[key] for w in all_wfo]
            return float(np.mean(vals)) if vals else 0.0

        rules = _describe_path_a_rules(condition_code, direction, indicator_filters, params)
        m.save_algo_result(
            db_path, session_id, None, rank,
            f"PathA_Strategy_#{rank}",
            params, rules,
            {
                "is_return":    avg("is_return"),
                "oos_return":   avg("oos_return"),
                "win_rate":     avg("oos_win_rate"),
                "sharpe":       avg("oos_sharpe"),
                "max_drawdown": avg("oos_max_dd"),
            },
        )

    progress(n_trials, n_trials, f"Path A complete: {len(top5)} strategies found.")
    m.update_task(db_path, task_id, result={"top_count": len(top5), "mode": "path_a"})


def _describe_path_a_rules(condition_code: str, direction: str,
                            indicator_filters: list, params: dict) -> str:
    lines = [
        f"MODE: Path A ({direction.upper()})", "",
        "BASE ENTRY CONDITION:",
        f"  {condition_code}", "",
        "TUNED FILTER CONDITIONS (Optuna):",
    ]
    for f in indicator_filters:
        col = f["col"]
        op  = f["operator"]
        threshold = params.get(f"threshold_{col}", f.get("threshold", "?"))
        if isinstance(threshold, float):
            threshold = f"{threshold:.4g}"
        lines.append(
            f"  • {col} {op} {threshold}  [Cohen's D={f.get('cohens_d', '?')}]"
        )
    lines += [
        "", "EXIT CONDITIONS:",
        f"  • RSI({int(params.get('exit_rsi_period', 14))}) > "
        f"{params.get('exit_rsi_threshold', 70):.1f}",
        "  • EMA(20) crosses below EMA(50)",
    ]
    return "\n".join(lines)
