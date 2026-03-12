"""
Phase 8: Algo Finder service.
Uses Optuna to search for the best combination of indicator rules
that maximizes profitability.

Two modes:
  Path B (default): Free search across the full indicator space.
  Path A: User-defined base entry logic is fixed; Optuna only tunes
          filter thresholds identified by the winner/loser analysis.
"""
import logging
import os
import re
import threading
import uuid
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


SEARCH_SPACE = {
    "rsi_period": (7, 21),
    "rsi_entry_max": (20, 50),
    "rsi_exit_min": (60, 85),
    "ema_fast": (8, 30),
    "ema_slow": (20, 100),
    "adx_min": (15, 35),
    "macd_hist_positive": (0, 1),     # 0 or 1 (bool)
    "bb_pct_entry_max": (0.1, 0.4),
    "volume_ratio_min": (0.5, 2.5),
    "use_supertrend": (0, 1),         # 0 or 1
    # ── Multi-Timeframe filters (applied when HTF data is present) ──
    "use_htf_trend": (0, 1),          # require HTF price > HTF EMA50 (trend_dir == 1)
    "use_htf_supertrend": (0, 1),     # require HTF Supertrend bullish
    "htf_rsi_max": (30, 70),          # optional HTF RSI max threshold
    "use_htf_rsi_filter": (0, 1),     # whether to apply htf_rsi_max filter
    "htf_adx_min": (15, 40),          # optional HTF ADX minimum
    "use_htf_adx_filter": (0, 1),     # whether to apply htf_adx_min filter
}


class OptunaStrategy(BaseStrategy):
    """Strategy parameterized by Optuna trial."""

    name = "OptunaStrategy"

    def __init__(self, params: dict):
        super().__init__(params)

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rsi_col = f"RSI_{int(p.get('rsi_period', 14))}"
        ema_fast_col = f"EMA_{int(p.get('ema_fast', 20))}"
        ema_slow_col = f"EMA_{int(p.get('ema_slow', 50))}"

        cond = pd.Series(True, index=df.index)

        # ── Primary TF filters ────────────────────────────────────────────────
        if rsi_col in df.columns:
            cond &= df[rsi_col] < p.get("rsi_entry_max", 40)

        if ema_fast_col in df.columns and ema_slow_col in df.columns:
            cond &= df[ema_fast_col] > df[ema_slow_col]

        if "ADX_14" in df.columns:
            cond &= df["ADX_14"] > p.get("adx_min", 20)

        if p.get("macd_hist_positive", 1) and "MACD_hist" in df.columns:
            cond &= df["MACD_hist"] > 0

        if "BB_pct_20" in df.columns:
            cond &= df["BB_pct_20"] < p.get("bb_pct_entry_max", 0.3)

        if "volume_ratio" in df.columns:
            cond &= df["volume_ratio"] > p.get("volume_ratio_min", 1.0)

        if p.get("use_supertrend", 0) and "SUPERT_dir" in df.columns:
            cond &= df["SUPERT_dir"] == 1

        # ── Multi-Timeframe filters ───────────────────────────────────────────
        # Find any HTF trend_dir column present (e.g. HTF_1h_trend_dir, HTF_4h_trend_dir)
        htf_trend_cols = [c for c in df.columns if c.endswith("_trend_dir")]
        if p.get("use_htf_trend", 0) and htf_trend_cols:
            # Use the lowest available higher TF trend direction
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
        rsi_col = f"RSI_{int(p.get('rsi_period', 14))}"

        cond = pd.Series(False, index=df.index)

        if rsi_col in df.columns:
            cond |= df[rsi_col] > p.get("rsi_exit_min", 70)

        if "ema_20_50_cross" in df.columns:
            cond |= df["ema_20_50_cross"] == -1

        df["exit_signal"] = cond.astype(int)
        return df


def _objective(trial, dfs: list, config: dict, has_htf: bool = False) -> float:
    """Optuna objective: returns OOS Sharpe ratio (maximize)."""
    params = {
        "rsi_period": trial.suggest_int("rsi_period", *SEARCH_SPACE["rsi_period"]),
        "rsi_entry_max": trial.suggest_float("rsi_entry_max", *SEARCH_SPACE["rsi_entry_max"]),
        "rsi_exit_min": trial.suggest_float("rsi_exit_min", *SEARCH_SPACE["rsi_exit_min"]),
        "ema_fast": trial.suggest_int("ema_fast", *SEARCH_SPACE["ema_fast"]),
        "ema_slow": trial.suggest_int("ema_slow", *SEARCH_SPACE["ema_slow"]),
        "adx_min": trial.suggest_float("adx_min", *SEARCH_SPACE["adx_min"]),
        "macd_hist_positive": trial.suggest_categorical("macd_hist_positive", [0, 1]),
        "bb_pct_entry_max": trial.suggest_float("bb_pct_entry_max", *SEARCH_SPACE["bb_pct_entry_max"]),
        "volume_ratio_min": trial.suggest_float("volume_ratio_min", *SEARCH_SPACE["volume_ratio_min"]),
        "use_supertrend": trial.suggest_categorical("use_supertrend", [0, 1]),
    }
    # HTF filters: only sampled if HTF data is present
    if has_htf:
        params["use_htf_trend"]      = trial.suggest_categorical("use_htf_trend", [0, 1])
        params["use_htf_supertrend"] = trial.suggest_categorical("use_htf_supertrend", [0, 1])
        params["use_htf_rsi_filter"] = trial.suggest_categorical("use_htf_rsi_filter", [0, 1])
        params["htf_rsi_max"]        = trial.suggest_float("htf_rsi_max", *SEARCH_SPACE["htf_rsi_max"])
        params["use_htf_adx_filter"] = trial.suggest_categorical("use_htf_adx_filter", [0, 1])
        params["htf_adx_min"]        = trial.suggest_float("htf_adx_min", *SEARCH_SPACE["htf_adx_min"])

    # Ensure fast EMA < slow EMA
    if params["ema_fast"] >= params["ema_slow"]:
        return -999.0

    strategy = OptunaStrategy(params)
    all_wfo = []

    for df in dfs:
        if len(df) < 100:
            continue
        # df already has all indicators pre-computed — skip populate_indicators()
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
            fast_mode=True,  # skip indicator recomputation and equity curve per fold
        )
        all_wfo.append(wfo)

    if not all_wfo:
        return -999.0

    # Objective: OOS Sharpe - penalize low trade count
    oos_sharpe = np.mean([w["oos_sharpe"] for w in all_wfo])
    oos_trades = np.mean([w["oos_trades"] for w in all_wfo])
    oos_return = np.mean([w["oos_return"] for w in all_wfo])

    if oos_trades < 5:
        return -999.0

    # Combined score: sharpe + return bonus
    score = oos_sharpe + oos_return * 0.01
    return float(score)


def run_algofinder(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list,
                   n_trials: int = 50, config: dict = None) -> None:
    """
    Background task: run Optuna search for best strategy parameters.
    """
    if config is None:
        config = {}

    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    progress(0, n_trials, "Loading data for Algo Finder…")

    pairs = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    primary_tf = timeframes[0] if timeframes else "1h"
    higher_tfs = timeframes[1:] if len(timeframes) > 1 else []

    # Load a sample of pairs (up to 20 for performance) — parallel
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
    workers = min(os.cpu_count() or 4, len(sample_pairs))
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = [exe.submit(_load_pair, pair) for pair in sample_pairs]
        for fut in as_completed(futs):
            result = fut.result()
            if result is not None:
                dfs.append(result)

    htf_found = htf_found_flag[0]

    if not dfs:
        m.update_task(db_path, task_id, status="error",
                      error="No data available for Algo Finder.",
                      message="Please download and enrich data first.")
        return

    tf_desc = f"{primary_tf}" + (f" + HTF: {', '.join(higher_tfs)}" if htf_found else " (no HTF data)")
    progress(0, n_trials, f"Running {n_trials} trials on {len(dfs)} pairs [{tf_desc}]…")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

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
            progress(count, n_trials, f"Trial {count}/{n_trials} — best: {best:.3f}")

    n_jobs = min(os.cpu_count() or 1, 4)  # up to 4 parallel Optuna workers
    study.optimize(
        lambda trial: _objective(trial, dfs, config, has_htf=htf_found),
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[callback],
        show_progress_bar=False,
    )

    # Get top-5 trials
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value > -100]
    completed.sort(key=lambda t: t.value, reverse=True)
    top5 = completed[:5]

    for rank, trial in enumerate(top5, 1):
        params = trial.params
        strategy = OptunaStrategy(params)

        # Compute final metrics on all data (df already has indicators pre-computed)
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
            return float(np.mean(vals)) if vals else 0

        rules = _describe_rules(params)
        m.save_algo_result(
            db_path, session_id, None, rank,
            f"AlgoStrategy_#{rank}",
            params,
            rules,
            {
                "is_return": avg("is_return"),
                "oos_return": avg("oos_return"),
                "win_rate": avg("oos_win_rate"),
                "sharpe": avg("oos_sharpe"),
                "max_drawdown": avg("oos_max_dd"),
            }
        )

    progress(n_trials, n_trials,
             f"Algo Finder complete: {len(top5)} strategies found.")
    m.update_task(db_path, task_id, result={"top_count": len(top5)})


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
        # indicator_filters: list of {col, operator} from top_indicators
        self._indicator_filters = indicator_filters

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # Evaluate the base user entry condition
        base_signal = _eval_entry_condition(df, self._condition_code)

        # Apply tunable filters on top
        cond = base_signal.copy()
        for f in self._indicator_filters:
            col = f["col"]
            op = f["operator"]
            param_key = f"threshold_{col}"
            threshold = self.params.get(param_key)
            if threshold is None or col not in df.columns:
                continue
            if op == ">":
                cond &= df[col] > threshold
            elif op == "<":
                cond &= df[col] < threshold

        df["entry_signal"] = cond.astype(int)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # Default exit: RSI overbought or EMA cross
        rsi_period = int(self.params.get("exit_rsi_period", 14))
        rsi_col = f"RSI_{rsi_period}"
        rsi_exit = float(self.params.get("exit_rsi_threshold", 70))

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
        col = f["col"]
        op = f["operator"]
        w_p25 = f.get("winner_p25", 0.0)
        w_p75 = f.get("winner_p75", 1.0)
        w_mean = f.get("winner_mean", (w_p25 + w_p75) / 2)

        # Search range: ±50% around winner IQR
        lo = min(w_p25, w_mean) * 0.5
        hi = max(w_p75, w_mean) * 1.5
        if op == "<":
            lo, hi = hi * 0.3, hi * 1.5  # flip the search range for < operator

        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        if abs(hi - lo) < 1e-8:
            hi = lo + 1.0

        params[f"threshold_{col}"] = trial.suggest_float(f"threshold_{col}", lo, hi)

    # Exit parameters
    params["exit_rsi_period"] = trial.suggest_categorical("exit_rsi_period", [7, 14, 21])
    params["exit_rsi_threshold"] = trial.suggest_float("exit_rsi_threshold", 60, 85)

    strategy = PathAStrategy(params, condition_code, indicator_filters)
    all_wfo = []

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
    top_indicators = analysis_result.get("top_indicators", [])

    # Use top 5 discriminative indicators as filters
    indicator_filters = top_indicators[:5]

    if not indicator_filters:
        m.update_task(db_path, task_id, status="error",
                      error="No indicator filters from analysis. Run Entry Logic Analysis first.",
                      message="Go to Entry Logic → Run Analysis first.")
        return

    progress(0, n_trials, f"Loading data for Path A Algo Finder ({len(indicator_filters)} filters)…")

    pairs = m.list_pairs(db_path, session_id)
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

    sample_pairs_a = active[:20]
    workers_a = min(os.cpu_count() or 4, len(sample_pairs_a))
    with ThreadPoolExecutor(max_workers=workers_a) as exe:
        futs = [exe.submit(_load_pair_a, pair) for pair in sample_pairs_a]
        for fut in as_completed(futs):
            result = fut.result()
            if result is not None:
                dfs.append(result)

    if not dfs:
        m.update_task(db_path, task_id, status="error",
                      error="No data available.", message="Download and add indicators first.")
        return

    progress(0, n_trials, f"Running {n_trials} Path A trials on {len(dfs)} pairs…")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
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
            progress(count, n_trials,
                     f"Trial {count}/{n_trials} — best: {best:.3f}")

    n_jobs_a = min(os.cpu_count() or 1, 4)
    study.optimize(
        lambda trial: _path_a_objective(trial, dfs, config, condition_code, indicator_filters),
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
        params = trial.params
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
            return float(np.mean(vals)) if vals else 0

        rules = _describe_path_a_rules(condition_code, direction, indicator_filters, params)
        m.save_algo_result(
            db_path, session_id, None, rank,
            f"PathA_Strategy_#{rank}",
            params, rules,
            {
                "is_return": avg("is_return"),
                "oos_return": avg("oos_return"),
                "win_rate": avg("oos_win_rate"),
                "sharpe": avg("oos_sharpe"),
                "max_drawdown": avg("oos_max_dd"),
            }
        )

    progress(n_trials, n_trials,
             f"Path A complete: {len(top5)} strategies found.")
    m.update_task(db_path, task_id, result={"top_count": len(top5), "mode": "path_a"})


def _describe_path_a_rules(condition_code: str, direction: str,
                            indicator_filters: list, params: dict) -> str:
    lines = [
        f"MODE: Path A ({direction.upper()})",
        f"",
        f"BASE ENTRY CONDITION:",
        f"  {condition_code}",
        f"",
        f"TUNED FILTER CONDITIONS (Optuna):",
    ]
    for f in indicator_filters:
        col = f["col"]
        op = f["operator"]
        threshold = params.get(f"threshold_{col}", f.get("threshold", "?"))
        if isinstance(threshold, float):
            threshold = f"{threshold:.4g}"
        lines.append(f"  • {col} {op} {threshold}  [Cohen's D={f.get('cohens_d', '?')}]")
    lines.extend([
        f"",
        f"EXIT CONDITIONS:",
        f"  • RSI({int(params.get('exit_rsi_period', 14))}) > {params.get('exit_rsi_threshold', 70):.1f}",
        f"  • EMA(20) crosses below EMA(50)",
    ])
    return "\n".join(lines)


def _describe_rules(params: dict) -> str:
    """Convert params dict to human-readable rule description."""
    lines = [
        f"ENTRY CONDITIONS:",
        f"  • RSI({int(params.get('rsi_period', 14))}) < {params.get('rsi_entry_max', 40):.1f}",
        f"  • EMA({int(params.get('ema_fast', 20))}) > EMA({int(params.get('ema_slow', 50))})",
        f"  • ADX(14) > {params.get('adx_min', 20):.1f}",
    ]
    if params.get("macd_hist_positive"):
        lines.append("  • MACD Histogram > 0")
    if params.get("bb_pct_entry_max", 1) < 0.99:
        lines.append(f"  • BB%B < {params.get('bb_pct_entry_max', 0.3):.2f} (near lower band)")
    if params.get("volume_ratio_min", 0) > 0.6:
        lines.append(f"  • Volume Ratio > {params.get('volume_ratio_min', 1):.1f}×")
    if params.get("use_supertrend"):
        lines.append("  • Supertrend direction = Bullish")
    # HTF filters
    if params.get("use_htf_trend"):
        lines.append("  • [HTF] Price > HTF EMA50 (trend aligned)")
    if params.get("use_htf_supertrend"):
        lines.append("  • [HTF] Supertrend direction = Bullish on higher TF")
    if params.get("use_htf_rsi_filter"):
        lines.append(f"  • [HTF] RSI < {params.get('htf_rsi_max', 60):.1f} on higher TF")
    if params.get("use_htf_adx_filter"):
        lines.append(f"  • [HTF] ADX > {params.get('htf_adx_min', 20):.1f} on higher TF")
    lines.extend([
        f"",
        f"EXIT CONDITIONS:",
        f"  • RSI({int(params.get('rsi_period', 14))}) > {params.get('rsi_exit_min', 70):.1f}",
        f"  • EMA({int(params.get('ema_fast', 20))}) crosses below EMA({int(params.get('ema_slow', 50))})",
    ])
    return "\n".join(lines)
