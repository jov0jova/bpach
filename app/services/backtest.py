"""
Full-fledged backtesting engine.

Features
────────
  • Fixed / ATR-based stop loss
  • Fixed / ATR-based / Risk:Reward take profit
  • Fixed / ATR-based trailing stop loss
  • Position sizing: fixed % | Kelly criterion | ATR-risk based
  • Dynamic pairlist (subset of session pairs)
  • Custom entry/exit logic via Python code editor
  • Walk-forward optimization (rolling or anchored windows)
  • Rich metrics: Sharpe, Sortino, Calmar, Expectancy, Recovery Factor,
    Profit Factor, per-pair breakdown
"""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy, CodeStrategy
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_df(parquet_dir, session_id, symbol, timeframe) -> Optional[pd.DataFrame]:
    path = parquet_path(parquet_dir, session_id, symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if len(df) >= 50 else None


def _kelly_position_size(win_rate: float, avg_win: float, avg_loss: float,
                          fraction: float = 0.25) -> float:
    """Fractional Kelly criterion position size."""
    if avg_loss <= 0 or win_rate <= 0:
        return 0.05
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b
    return float(np.clip(kelly * fraction, 0.01, 0.5))


def _compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder's smoothed ATR."""
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n  = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))
    atr = np.empty(n)
    atr[:period] = np.nan
    if period <= n:
        atr[period-1] = float(np.mean(tr[:period]))
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def _simple_backtest(df: pd.DataFrame,
                     initial_capital: float = 10_000,
                     fee_rate: float = 0.001,
                     slippage: float = 0.0005,
                     position_size: float = 0.1,
                     # Stop loss
                     sl_mode: str = "none",
                     sl_pct: float = 0.02,
                     sl_atr_period: int = 14,
                     sl_atr_multiplier: float = 2.0,
                     # Take profit
                     tp_mode: str = "none",
                     tp_pct: float = 0.04,
                     tp_atr_period: int = 14,
                     tp_atr_multiplier: float = 4.0,
                     tp_rr_ratio: float = 2.0,
                     # Trailing stop
                     trail_mode: str = "none",
                     trail_pct: float = 0.02,
                     trail_atr_period: int = 14,
                     trail_atr_multiplier: float = 1.5,
                     # Position sizing
                     position_sizing: str = "fixed",
                     kelly_fraction: float = 0.25,
                     atr_risk_pct: float = 0.01,
                     # Legacy shims
                     stop_loss_pct: float = 0.0,
                     take_profit_pct: float = 0.0,
                     trailing_stop_pct: float = 0.0,
                     fast_mode: bool = False) -> dict:
    """
    Pure-Python backtest engine.

    sl_mode:        'none' | 'fixed' | 'atr'
    tp_mode:        'none' | 'fixed' | 'atr' | 'rr'
    trail_mode:     'none' | 'fixed' | 'atr'
    position_sizing:'fixed' | 'kelly' | 'atr_risk'
    """
    # Legacy API: map old pct params to new API
    if stop_loss_pct > 0 and sl_mode == "none":
        sl_mode = "fixed"; sl_pct = stop_loss_pct
    if take_profit_pct > 0 and tp_mode == "none":
        tp_mode = "fixed"; tp_pct = take_profit_pct
    if trailing_stop_pct > 0 and trail_mode == "none":
        trail_mode = "fixed"; trail_pct = trailing_stop_pct

    # Pre-compute ATR arrays
    atr_periods_needed = set()
    if sl_mode    == "atr":      atr_periods_needed.add(sl_atr_period)
    if tp_mode    == "atr":      atr_periods_needed.add(tp_atr_period)
    if trail_mode == "atr":      atr_periods_needed.add(trail_atr_period)
    if position_sizing == "atr_risk": atr_periods_needed.add(14)

    atr_cache: dict[int, np.ndarray] = {}
    for p in atr_periods_needed:
        key = f"ATR_{p}"
        atr_cache[p] = df[key].values if key in df.columns else _compute_atr(df, p)

    def _atr(period: int, i: int) -> float:
        arr = atr_cache.get(period)
        if arr is None or i >= len(arr):
            return 0.0
        v = arr[i]
        return float(v) if np.isfinite(v) else 0.0

    capital   = initial_capital
    equity    = [] if fast_mode else [capital]
    in_trade  = False
    entry_price = tp_price = sl_price = trail_dist = cur_pos = 0.0
    entry_idx = 0
    peak_price = 0.0
    kelly_pos  = position_size
    trades     = []

    entry_sig = df["entry_signal"].fillna(0).values
    exit_sig  = (df["exit_signal"].fillna(0).values
                 if "exit_signal" in df.columns else np.zeros(len(df)))
    close = df["close"].values
    ts    = df["timestamp"].values
    n     = len(close)

    for i in range(1, n):
        if not in_trade:
            if entry_sig[i - 1] != 1:
                if not fast_mode:
                    equity.append(capital)
                continue

            # ── ENTER ────────────────────────────────────────────────────────
            ep          = close[i] * (1 + slippage)
            in_trade    = True
            entry_price = ep
            peak_price  = ep
            entry_idx   = i

            if sl_mode == "atr":
                av = _atr(sl_atr_period, i)
                sl_price = ep - av * sl_atr_multiplier if av > 0 else ep * (1 - sl_pct)
            elif sl_mode == "fixed":
                sl_price = ep * (1 - sl_pct)
            else:
                sl_price = 0.0

            if tp_mode == "atr":
                av = _atr(tp_atr_period, i)
                tp_price = ep + av * tp_atr_multiplier if av > 0 else ep * (1 + tp_pct)
            elif tp_mode == "fixed":
                tp_price = ep * (1 + tp_pct)
            elif tp_mode == "rr" and sl_price > 0:
                tp_price = ep + (ep - sl_price) * tp_rr_ratio
            else:
                tp_price = 0.0

            if trail_mode == "atr":
                av = _atr(trail_atr_period, i)
                trail_dist = av * trail_atr_multiplier if av > 0 else ep * trail_pct
            elif trail_mode == "fixed":
                trail_dist = ep * trail_pct
            else:
                trail_dist = 0.0

            if position_sizing == "kelly":
                cur_pos = kelly_pos
            elif position_sizing == "atr_risk":
                av = _atr(14, i)
                cur_pos = min(capital * atr_risk_pct / (av * ep / ep), 0.5) if av > 0 else position_size
            else:
                cur_pos = position_size

            capital -= capital * cur_pos * fee_rate

        else:
            # ── IN TRADE ─────────────────────────────────────────────────────
            cp         = close[i]
            peak_price = max(peak_price, cp)

            exit_reason = None
            if sl_price > 0 and cp <= sl_price:
                exit_reason = "stop_loss"
            elif tp_price > 0 and cp >= tp_price:
                exit_reason = "take_profit"
            elif trail_dist > 0 and (peak_price - cp) >= trail_dist:
                exit_reason = "trailing_stop"
                if trail_mode == "atr":
                    nd = _atr(trail_atr_period, i) * trail_atr_multiplier
                    if nd > 0:
                        trail_dist = nd
            elif exit_sig[i - 1] == 1:
                exit_reason = "exit_signal"
            elif i == n - 1:
                exit_reason = "end_of_data"

            if exit_reason:
                xp      = cp * (1 - slippage)
                tr_ret  = (xp / entry_price - 1) * cur_pos
                capital = capital * (1 + tr_ret)
                capital -= capital * cur_pos * fee_rate
                pnl     = (xp / entry_price - 1) * 100
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": xp,
                    "pnl_pct": pnl, "is_winner": pnl > 0,
                    "entry_time": str(ts[entry_idx]), "exit_time": str(ts[i]),
                    "duration_bars": i - entry_idx,
                    "exit_reason": exit_reason,
                    "position_size": cur_pos,
                })
                in_trade = False
                # Update Kelly estimate every 10 trades
                if len(trades) % 10 == 0 and len(trades) >= 10:
                    rec = trades[-20:]
                    wns = [t for t in rec if t["is_winner"]]
                    lss = [t for t in rec if not t["is_winner"]]
                    if wns and lss:
                        kelly_pos = _kelly_position_size(
                            len(wns) / len(rec),
                            float(np.mean([t["pnl_pct"] for t in wns])),
                            float(abs(np.mean([t["pnl_pct"] for t in lss]))),
                            kelly_fraction)

        if not fast_mode:
            equity.append(capital)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0,
            "expectancy": 0, "recovery_factor": 0,
            "max_drawdown": 0, "total_return": 0, "avg_trade_duration": 0,
            "equity": equity,
            "timestamps": [] if fast_mode else df["timestamp"].astype(str).tolist(),
            "trades_list": [],
        }

    winners  = [t for t in trades if t["is_winner"]]
    losers   = [t for t in trades if not t["is_winner"]]
    wr       = len(winners) / len(trades)
    gp       = sum(t["pnl_pct"] for t in winners) if winners else 0
    gl       = abs(sum(t["pnl_pct"] for t in losers)) if losers else 1e-9
    pf       = gp / gl
    avg_w    = float(np.mean([t["pnl_pct"] for t in winners])) if winners else 0
    avg_l    = float(abs(np.mean([t["pnl_pct"] for t in losers]))) if losers else 0
    expy     = (wr * avg_w) - ((1 - wr) * avg_l)
    tot_ret  = (capital / initial_capital - 1) * 100
    avg_dur  = float(np.mean([t["duration_bars"] for t in trades]))

    if fast_mode:
        trets   = np.array([t["pnl_pct"] / 100 * t["position_size"] for t in trades])
        r_std   = float(np.std(trets)) + 1e-9
        d_std   = float(np.std([r for r in trets if r < 0])) + 1e-9
        sharpe  = float(np.mean(trets) / r_std  * np.sqrt(252))
        sortino = float(np.mean(trets) / d_std  * np.sqrt(252))
        max_dd = calmar = recovery = 0.0
    else:
        eq_arr  = np.array(equity)
        peaks   = np.maximum.accumulate(eq_arr)
        dds     = (peaks - eq_arr) / (peaks + 1e-12) * 100
        max_dd  = float(np.max(dds))
        rets    = np.diff(eq_arr) / (eq_arr[:-1] + 1e-12)
        bpy     = 252 * 24
        d_rets  = np.array([r for r in rets if r < 0])
        d_std   = float(np.std(d_rets)) + 1e-9 if len(d_rets) > 0 else 1e-9
        m_r     = float(np.mean(rets))
        s_r     = float(np.std(rets)) + 1e-9
        sharpe  = float(m_r / s_r  * np.sqrt(bpy))
        sortino = float(m_r / d_std * np.sqrt(bpy))
        calmar  = tot_ret / max_dd if max_dd > 0 else 0.0
        recovery = abs(tot_ret) / max_dd if max_dd > 0 else 0.0

    return {
        "total_trades":      len(trades),
        "win_rate":          wr,
        "profit_factor":     pf,
        "sharpe_ratio":      sharpe,
        "sortino_ratio":     sortino,
        "calmar_ratio":      calmar,
        "expectancy":        expy,
        "recovery_factor":   recovery,
        "max_drawdown":      max_dd,
        "total_return":      tot_ret,
        "avg_trade_duration": avg_dur,
        "equity":            equity,
        "timestamps":        [] if fast_mode else df["timestamp"].astype(str).tolist(),
        "trades_list":       trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_backtest(df: pd.DataFrame,
                           strategy,
                           n_splits: int,
                           train_ratio: float,
                           bt_kwargs: dict,
                           fast_mode: bool = False,
                           anchored: bool = False) -> dict:
    """
    Walk-forward validation.

    anchored=True  → expanding train window from bar 0.
    anchored=False → rolling fixed-size windows.
    """
    n         = len(df)
    fold_size = n // n_splits
    is_list  = []
    oos_list = []

    for fold in range(n_splits):
        if anchored:
            end      = (fold + 1) * fold_size if fold < n_splits - 1 else n
            split    = fold * fold_size + int((end - fold * fold_size) * train_ratio)
            train_df = df.iloc[:split].copy()
            test_df  = df.iloc[split:end].copy()
        else:
            start    = fold * fold_size
            end      = start + fold_size if fold < n_splits - 1 else n
            fdf      = df.iloc[start:end].copy()
            split    = int(len(fdf) * train_ratio)
            train_df = fdf.iloc[:split].copy()
            test_df  = fdf.iloc[split:].copy()

        if len(train_df) < 50 or len(test_df) < 20:
            continue

        if fast_mode:
            tr = strategy.populate_entry_signal(train_df)
            tr = strategy.populate_exit_signal(tr)
            te = strategy.populate_entry_signal(test_df)
            te = strategy.populate_exit_signal(te)
        else:
            tr = strategy.run(train_df)
            te = strategy.run(test_df)

        kw = {**bt_kwargs, "fast_mode": fast_mode}
        is_list.append(_simple_backtest(tr, **kw))
        oos_list.append(_simple_backtest(te, **kw))

    def _avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "is_return":         _avg(is_list,  "total_return"),
        "oos_return":        _avg(oos_list, "total_return"),
        "is_win_rate":       _avg(is_list,  "win_rate"),
        "oos_win_rate":      _avg(oos_list, "win_rate"),
        "is_sharpe":         _avg(is_list,  "sharpe_ratio"),
        "oos_sharpe":        _avg(oos_list, "sharpe_ratio"),
        "is_sortino":        _avg(is_list,  "sortino_ratio"),
        "oos_sortino":       _avg(oos_list, "sortino_ratio"),
        "is_calmar":         _avg(is_list,  "calmar_ratio"),
        "oos_calmar":        _avg(oos_list, "calmar_ratio"),
        "is_max_dd":         _avg(is_list,  "max_drawdown"),
        "oos_max_dd":        _avg(oos_list, "max_drawdown"),
        "is_trades":         int(sum(r.get("total_trades",0) for r in is_list)),
        "oos_trades":        int(sum(r.get("total_trades",0) for r in oos_list)),
        "is_profit_factor":  _avg(is_list,  "profit_factor"),
        "oos_profit_factor": _avg(oos_list, "profit_factor"),
        "oos_expectancy":    _avg(oos_list, "expectancy"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Background task entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(task_id: str, db_path: Path, session_id: str,
                 parquet_dir: Path, run_id: str,
                 strategy_code: str, params: dict,
                 timeframes: list, config: dict,
                 selected_pairs: list = None,
                 stop_event=None) -> None:
    """
    Background task: full-fledged backtest on selected (or all) pairs.

    config keys:
        initial_capital, fee_rate, slippage,
        position_sizing, position_size, kelly_fraction, atr_risk_pct,
        sl_mode, sl_pct, sl_atr_period, sl_atr_multiplier,
        tp_mode, tp_pct, tp_atr_period, tp_atr_multiplier, tp_rr_ratio,
        trail_mode, trail_pct, trail_atr_period, trail_atr_multiplier,
        wfo_splits, wfo_train_ratio, wfo_anchored
    """
    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    initial_capital  = float(config.get("initial_capital", 10_000))
    fee_rate         = float(config.get("fee_rate", 0.001))
    slippage         = float(config.get("slippage", 0.0005))
    position_sizing  = config.get("position_sizing", "fixed")
    position_size    = float(config.get("position_size", 0.1))
    kelly_fraction   = float(config.get("kelly_fraction", 0.25))
    atr_risk_pct     = float(config.get("atr_risk_pct", 0.01))
    sl_mode          = config.get("sl_mode", "none")
    sl_pct           = float(config.get("sl_pct", 0.02))
    sl_atr_period    = int(config.get("sl_atr_period", 14))
    sl_atr_mult      = float(config.get("sl_atr_multiplier", 2.0))
    tp_mode          = config.get("tp_mode", "none")
    tp_pct           = float(config.get("tp_pct", 0.04))
    tp_atr_period    = int(config.get("tp_atr_period", 14))
    tp_atr_mult      = float(config.get("tp_atr_multiplier", 4.0))
    tp_rr_ratio      = float(config.get("tp_rr_ratio", 2.0))
    trail_mode       = config.get("trail_mode", "none")
    trail_pct        = float(config.get("trail_pct", 0.02))
    trail_atr_period = int(config.get("trail_atr_period", 14))
    trail_atr_mult   = float(config.get("trail_atr_multiplier", 1.5))
    n_splits         = int(config.get("wfo_splits", 5))
    train_ratio      = float(config.get("wfo_train_ratio", 0.7))
    wfo_anchored     = bool(config.get("wfo_anchored", False))
    primary_tf       = timeframes[0] if timeframes else "1h"

    bt_kwargs = dict(
        initial_capital=initial_capital, fee_rate=fee_rate, slippage=slippage,
        position_size=position_size, position_sizing=position_sizing,
        kelly_fraction=kelly_fraction, atr_risk_pct=atr_risk_pct,
        sl_mode=sl_mode, sl_pct=sl_pct,
        sl_atr_period=sl_atr_period, sl_atr_multiplier=sl_atr_mult,
        tp_mode=tp_mode, tp_pct=tp_pct,
        tp_atr_period=tp_atr_period, tp_atr_multiplier=tp_atr_mult,
        tp_rr_ratio=tp_rr_ratio,
        trail_mode=trail_mode, trail_pct=trail_pct,
        trail_atr_period=trail_atr_period, trail_atr_multiplier=trail_atr_mult,
    )

    all_pairs = m.list_pairs(db_path, session_id)
    if selected_pairs:
        active = [p for p in all_pairs
                  if p["symbol"] in selected_pairs and p["candle_count"] > 0]
    else:
        active = [p for p in all_pairs
                  if not p["excluded"] and p["candle_count"] > 0]

    if not active:
        m.update_task(db_path, task_id, status="error",
                      error="No active pairs with data.", message="Download data first.")
        m.update_backtest_run(db_path, run_id, status="error", error_msg="No active pairs.")
        return

    progress(0, len(active), "Building strategy…")
    m.update_backtest_run(db_path, run_id, status="running")

    try:
        strategy = (CodeStrategy(strategy_code, params)
                    if strategy_code and strategy_code.strip()
                    else BaseStrategy(params))
    except Exception as e:
        m.update_backtest_run(db_path, run_id, status="error", error_msg=str(e))
        m.update_task(db_path, task_id, status="error", error=str(e))
        return

    all_trades       = []
    all_wfo          = []
    pair_results     = []
    pairs_profitable = 0

    for idx, pair in enumerate(active):
        if stop_event and stop_event.is_set():
            break
        symbol = pair["symbol"]
        progress(idx, len(active), f"Backtesting {symbol}…")

        df = _load_df(parquet_dir, session_id, symbol, primary_tf)
        if df is None:
            continue
        try:
            df = strategy.run(df)
        except Exception as e:
            logger.warning("Strategy failed on %s: %s", symbol, e)
            continue
        if "entry_signal" not in df.columns or len(df) < 100:
            continue

        full = _simple_backtest(df, **bt_kwargs)

        try:
            wfo = _walk_forward_backtest(
                df, strategy, n_splits, train_ratio,
                bt_kwargs, fast_mode=True, anchored=wfo_anchored)
            all_wfo.append(wfo)
        except Exception as e:
            logger.warning("WFO failed on %s: %s", symbol, e)

        if full.get("total_return", 0) > 0:
            pairs_profitable += 1

        pair_results.append({
            "symbol":        symbol,
            "total_return":  full.get("total_return", 0),
            "total_trades":  full.get("total_trades", 0),
            "win_rate":      full.get("win_rate", 0),
            "sharpe":        full.get("sharpe_ratio", 0),
            "sortino":       full.get("sortino_ratio", 0),
            "calmar":        full.get("calmar_ratio", 0),
            "max_dd":        full.get("max_drawdown", 0),
            "profit_factor": full.get("profit_factor", 0),
            "expectancy":    full.get("expectancy", 0),
        })

        for t in full.get("trades_list", []):
            all_trades.append({
                "id": str(uuid.uuid4()),
                "run_id": run_id, "session_id": session_id,
                "symbol": symbol, "timeframe": primary_tf,
                "entry_time": t["entry_time"], "exit_time": t["exit_time"],
                "entry_price": t["entry_price"], "exit_price": t["exit_price"],
                "direction": "long",
                "pnl_pct": t["pnl_pct"],
                "pnl_abs": t["pnl_pct"] / 100 * initial_capital * t["position_size"],
                "duration_bars": t["duration_bars"],
                "is_winner": t["is_winner"],
                "entry_signals": {},
                "exit_reason": t["exit_reason"],
            })

    m.insert_trades(db_path, all_trades)

    def _avg(lst, key):
        vals = [w[key] for w in lst if key in w and w[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    total   = len(all_trades)
    winners = [t for t in all_trades if t["is_winner"]]
    losers  = [t for t in all_trades if not t["is_winner"]]
    wr      = len(winners) / total if total > 0 else 0
    gp      = sum(t["pnl_pct"] for t in winners) if winners else 0
    gl      = abs(sum(t["pnl_pct"] for t in losers)) if losers else 1e-9
    pf      = gp / gl
    avg_w   = float(np.mean([t["pnl_pct"] for t in winners])) if winners else 0
    avg_l   = float(abs(np.mean([t["pnl_pct"] for t in losers]))) if losers else 0
    expy    = float((wr * avg_w) - ((1 - wr) * avg_l))
    is_ret  = _avg(all_wfo, "is_return")
    oos_ret = _avg(all_wfo, "oos_return")
    max_dd  = _avg(all_wfo, "is_max_dd")
    sharpe  = _avg(all_wfo, "is_sharpe")
    sortino = _avg(all_wfo, "is_sortino")
    calmar  = is_ret / max_dd if max_dd > 0 else 0.0
    recovery = abs(is_ret) / max_dd if max_dd > 0 else 0.0
    avg_dur = float(np.mean([t["duration_bars"] for t in all_trades])) if all_trades else 0

    m.update_backtest_run(db_path, run_id,
        status="done",
        total_trades=total,
        win_rate=wr,
        profit_factor=pf,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_return=is_ret,
        avg_trade_duration=avg_dur,
        oos_return=oos_ret,
        oos_win_rate=_avg(all_wfo, "oos_win_rate"),
        completed_at=datetime.now(timezone.utc),
    )

    result_payload = {
        "run_id": run_id,
        "sortino": sortino, "calmar": calmar,
        "recovery_factor": recovery, "expectancy": expy,
        "profit_factor": pf,
        "pairs_backtested": len(pair_results),
        "pairs_profitable": pairs_profitable,
        "pair_coverage": pairs_profitable / len(pair_results) if pair_results else 0,
        "pair_results": pair_results,
        "oos_profit_factor": _avg(all_wfo, "oos_profit_factor"),
        "oos_expectancy": _avg(all_wfo, "oos_expectancy"),
    }
    progress(len(active), len(active),
             f"Done: {total} trades on {len(pair_results)} pairs | "
             f"WR={wr:.1%} PF={pf:.2f} Sharpe={sharpe:.2f} "
             f"IS={is_ret:+.1f}% OOS={oos_ret:+.1f}%")
    m.update_task(db_path, task_id, status="done", result=result_payload)
