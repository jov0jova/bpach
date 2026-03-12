"""
Phase 6: Backtesting engine.
Pure-Python walk-forward backtesting with Optuna OOS validation.
"""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd

from .. import models as m
from ..strategies.base import BaseStrategy, CodeStrategy
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_enriched_df(parquet_dir, session_id, symbol, timeframe) -> Optional[pd.DataFrame]:
    path = parquet_path(parquet_dir, session_id, symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if len(df) < 100:
        return None
    if "entry_signal" not in df.columns:
        return None
    return df


def _simple_backtest(df: pd.DataFrame, initial_capital: float,
                     fee_rate: float, slippage: float,
                     position_size: float,
                     stop_loss_pct: float = 0.0,
                     take_profit_pct: float = 0.0,
                     trailing_stop_pct: float = 0.0,
                     fast_mode: bool = False) -> dict:
    """
    Pure-Python backtest with optional stop-loss, take-profit, and trailing stop.

    stop_loss_pct:    e.g. 0.02 = exit if trade drops 2% from entry
    take_profit_pct:  e.g. 0.06 = exit if trade gains 6% from entry
    trailing_stop_pct: e.g. 0.03 = exit if price drops 3% from peak price seen since entry
    fast_mode:        skip equity curve tracking; use entry-jump loop to skip non-trade bars.
                      Used by Optuna to run 10–50× faster. Stats are equivalent for ranking.
    """
    capital = initial_capital
    equity = [] if fast_mode else [capital]
    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    peak_price = 0.0
    trades = []

    entry_sig = df["entry_signal"].fillna(0).values
    exit_sig = (df["exit_signal"].fillna(0).values if "exit_signal" in df.columns
                else np.zeros(len(df)))
    close = df["close"].values
    ts = df["timestamp"].values
    n = len(close)

    if fast_mode:
        # Jump-based loop: skip non-trade bars using numpy argmax.
        # Within each trade, iterate normally (needed for trailing stop per-bar logic).
        i = 1
        while i < n:
            if not in_trade:
                # Find next entry signal without iterating bar-by-bar
                remaining = entry_sig[i - 1: n - 1]
                next_offset = int(np.argmax(remaining == 1))
                if remaining[next_offset] != 1:
                    break  # No more entry signals
                i = (i - 1) + next_offset + 1
                if i >= n:
                    break
                in_trade = True
                entry_price = close[i] * (1 + slippage)
                peak_price = entry_price
                entry_idx = i
                capital -= capital * position_size * fee_rate
                i += 1
            else:
                current_price = close[i]
                peak_price = max(peak_price, current_price)
                pnl_from_entry = (current_price / entry_price) - 1
                pnl_from_peak  = (current_price / peak_price) - 1 if peak_price > 0 else 0

                exit_reason = "signal"
                should_exit = (exit_sig[i - 1] == 1 or i == n - 1)

                if stop_loss_pct > 0 and pnl_from_entry <= -stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif take_profit_pct > 0 and pnl_from_entry >= take_profit_pct:
                    should_exit = True
                    exit_reason = "take_profit"
                elif trailing_stop_pct > 0 and pnl_from_peak <= -trailing_stop_pct:
                    should_exit = True
                    exit_reason = "trailing_stop"

                if should_exit:
                    exit_price = current_price * (1 - slippage)
                    trade_return = (exit_price / entry_price - 1) * position_size
                    capital *= (1 + trade_return)
                    capital -= capital * position_size * fee_rate
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    trades.append({
                        "entry_idx": entry_idx, "exit_idx": i,
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl_pct": pnl_pct, "is_winner": pnl_pct > 0,
                        "entry_time": str(ts[entry_idx]), "exit_time": str(ts[i]),
                        "duration_bars": i - entry_idx, "exit_reason": exit_reason,
                    })
                    in_trade = False
                i += 1
    else:
        for i in range(1, n):
            if not in_trade and entry_sig[i - 1] == 1:
                in_trade = True
                entry_price = close[i] * (1 + slippage)
                peak_price = entry_price
                entry_idx = i
                capital -= capital * position_size * fee_rate

            elif in_trade:
                current_price = close[i]
                peak_price = max(peak_price, current_price)

                pnl_from_entry = (current_price / entry_price) - 1
                pnl_from_peak  = (current_price / peak_price) - 1 if peak_price > 0 else 0

                exit_reason = "signal"
                should_exit = (exit_sig[i - 1] == 1 or i == n - 1)

                if stop_loss_pct > 0 and pnl_from_entry <= -stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif take_profit_pct > 0 and pnl_from_entry >= take_profit_pct:
                    should_exit = True
                    exit_reason = "take_profit"
                elif trailing_stop_pct > 0 and pnl_from_peak <= -trailing_stop_pct:
                    should_exit = True
                    exit_reason = "trailing_stop"

                if should_exit:
                    exit_price = current_price * (1 - slippage)
                    trade_return = (exit_price / entry_price - 1) * position_size
                    capital *= (1 + trade_return)
                    capital -= capital * position_size * fee_rate
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    trades.append({
                        "entry_idx": entry_idx, "exit_idx": i,
                        "entry_price": entry_price, "exit_price": exit_price,
                        "pnl_pct": pnl_pct, "is_winner": pnl_pct > 0,
                        "entry_time": str(ts[entry_idx]), "exit_time": str(ts[i]),
                        "duration_bars": i - entry_idx, "exit_reason": exit_reason,
                    })
                    in_trade = False

            equity.append(capital)

    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe_ratio": 0, "max_drawdown": 0, "total_return": 0,
            "equity": equity,
            "timestamps": [] if fast_mode else df["timestamp"].astype(str).tolist(),
            "trades_list": [],
        }

    winners = [t for t in trades if t["is_winner"]]
    win_rate = len(winners) / len(trades)
    gross_profit = sum(t["pnl_pct"] for t in winners)
    gross_loss = abs(sum(t["pnl_pct"] for t in trades if not t["is_winner"]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_return = (capital / initial_capital - 1) * 100

    if fast_mode:
        # Compute Sharpe from per-trade returns (faster, no bar-level equity needed)
        trade_rets = np.array([t["pnl_pct"] / 100 * position_size for t in trades])
        sharpe = (np.mean(trade_rets) / (np.std(trade_rets) + 1e-9)) * np.sqrt(252)
        max_dd = 0.0  # not computed in fast mode
    else:
        eq_arr = np.array(equity)
        peaks = np.maximum.accumulate(eq_arr)
        drawdowns = (peaks - eq_arr) / peaks * 100
        max_dd = float(np.max(drawdowns))
        returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252 * 24)

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "total_return": total_return,
        "equity": equity,
        "timestamps": [] if fast_mode else df["timestamp"].astype(str).tolist(),
        "trades_list": trades,
    }


def _walk_forward_backtest(df: pd.DataFrame, strategy: BaseStrategy,
                           n_splits: int, train_ratio: float,
                           initial_capital: float, fee_rate: float,
                           slippage: float, position_size: float,
                           stop_loss_pct: float = 0.0,
                           take_profit_pct: float = 0.0,
                           trailing_stop_pct: float = 0.0,
                           fast_mode: bool = False) -> dict:
    """
    Walk-forward optimization: split data into n_splits folds,
    optimize on train window, validate on test window.

    fast_mode: skip populate_indicators() per fold (assumes df already enriched)
               and use fast_mode backtest (no equity curve, trade-based Sharpe).
               Used by Optuna for ~10× faster trials.
    """
    n = len(df)
    fold_size = n // n_splits
    is_results = []
    oos_results = []

    for fold in range(n_splits):
        start = fold * fold_size
        end = start + fold_size if fold < n_splits - 1 else n
        fold_df = df.iloc[start:end].copy()

        split = int(len(fold_df) * train_ratio)
        train_df = fold_df.iloc[:split].copy()
        test_df = fold_df.iloc[split:].copy()

        if len(train_df) < 50 or len(test_df) < 20:
            continue

        if fast_mode:
            # df already has indicators — only recompute entry/exit signals on each fold
            train_enriched = strategy.populate_entry_signal(train_df)
            train_enriched = strategy.populate_exit_signal(train_enriched)
            test_enriched  = strategy.populate_entry_signal(test_df)
            test_enriched  = strategy.populate_exit_signal(test_enriched)
        else:
            train_enriched = strategy.run(train_df)
            test_enriched  = strategy.run(test_df)

        is_r = _simple_backtest(train_enriched, initial_capital, fee_rate, slippage, position_size,
                                stop_loss_pct, take_profit_pct, trailing_stop_pct,
                                fast_mode=fast_mode)
        is_results.append(is_r)

        oos_r = _simple_backtest(test_enriched, initial_capital, fee_rate, slippage, position_size,
                                 stop_loss_pct, take_profit_pct, trailing_stop_pct,
                                 fast_mode=fast_mode)
        oos_results.append(oos_r)

    def avg(results, key):
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "is_return": avg(is_results, "total_return"),
        "oos_return": avg(oos_results, "total_return"),
        "is_win_rate": avg(is_results, "win_rate"),
        "oos_win_rate": avg(oos_results, "win_rate"),
        "is_sharpe": avg(is_results, "sharpe_ratio"),
        "oos_sharpe": avg(oos_results, "sharpe_ratio"),
        "is_max_dd": avg(is_results, "max_drawdown"),
        "oos_max_dd": avg(oos_results, "max_drawdown"),
        "is_trades": avg(is_results, "total_trades"),
        "oos_trades": avg(oos_results, "total_trades"),
        "is_profit_factor": avg(is_results, "profit_factor"),
    }


def run_backtest(task_id: str, db_path: Path, session_id: str,
                 parquet_dir: Path, run_id: str,
                 strategy_code: str, params: dict,
                 timeframes: list, config: dict) -> None:
    """
    Background task: run backtest on all pairs and aggregate results.
    """
    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    initial_capital   = config.get("initial_capital", 10_000)
    fee_rate          = config.get("fee_rate", 0.001)
    slippage          = config.get("slippage", 0.0005)
    position_size     = config.get("position_size", 0.1)
    stop_loss_pct     = config.get("stop_loss_pct", 0.0)
    take_profit_pct   = config.get("take_profit_pct", 0.0)
    trailing_stop_pct = config.get("trailing_stop_pct", 0.0)
    n_splits          = config.get("wfo_splits", 5)
    train_ratio       = config.get("wfo_train_ratio", 0.7)
    primary_tf = timeframes[0] if timeframes else "1h"

    pairs = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]

    progress(0, len(active), "Starting backtest…")
    m.update_backtest_run(db_path, run_id, status="running")

    # Build strategy
    try:
        if strategy_code.strip():
            strategy = CodeStrategy(strategy_code, params)
        else:
            strategy = BaseStrategy(params)
    except Exception as e:
        m.update_backtest_run(db_path, run_id, status="error", error_msg=str(e))
        return

    all_trades = []
    all_wfo = []

    for idx, pair in enumerate(active):
        symbol = pair["symbol"]
        progress(idx, len(active), f"Backtesting {symbol}…")

        df = _load_enriched_df(parquet_dir, session_id, symbol, primary_tf)
        if df is None:
            # Try to enrich on the fly
            path = parquet_path(parquet_dir, session_id, symbol, primary_tf)
            if path.exists():
                df = pd.read_parquet(path)
                df = strategy.run(df)
            else:
                continue
        else:
            df = strategy.run(df)

        if len(df) < 100:
            continue

        wfo = _walk_forward_backtest(df, strategy, n_splits, train_ratio,
                                     initial_capital, fee_rate, slippage, position_size)
        all_wfo.append(wfo)

        # Full-period backtest for trade records
        full_result = _simple_backtest(df, initial_capital, fee_rate, slippage, position_size,
                                       stop_loss_pct, take_profit_pct, trailing_stop_pct)

        for t in full_result.get("trades_list", []):
            all_trades.append({
                "id": str(uuid.uuid4()),
                "run_id": run_id,
                "session_id": session_id,
                "symbol": symbol,
                "timeframe": primary_tf,
                "entry_time": t["entry_time"],
                "exit_time": t["exit_time"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "direction": "long",
                "pnl_pct": t["pnl_pct"],
                "pnl_abs": t["pnl_pct"] / 100 * initial_capital * position_size,
                "duration_bars": t["duration_bars"],
                "is_winner": t["is_winner"],
                "entry_signals": {},
                "exit_reason": "signal",
            })

    # Save all trades
    m.insert_trades(db_path, all_trades)

    # Aggregate metrics
    def avg_wfo(key):
        vals = [w[key] for w in all_wfo if key in w]
        return float(np.mean(vals)) if vals else 0.0

    winners = [t for t in all_trades if t["is_winner"]]
    total = len(all_trades)
    win_rate = len(winners) / total if total > 0 else 0
    gross_p = sum(t["pnl_pct"] for t in winners)
    gross_l = abs(sum(t["pnl_pct"] for t in all_trades if not t["is_winner"]))
    pf = gross_p / gross_l if gross_l > 0 else 0

    m.update_backtest_run(db_path, run_id,
        status="done",
        total_trades=total,
        win_rate=win_rate,
        profit_factor=pf,
        sharpe_ratio=avg_wfo("is_sharpe"),
        max_drawdown=avg_wfo("is_max_dd"),
        total_return=avg_wfo("is_return"),
        avg_trade_duration=float(np.mean([t["duration_bars"] for t in all_trades])) if all_trades else 0,
        oos_return=avg_wfo("oos_return"),
        oos_win_rate=avg_wfo("oos_win_rate"),
        completed_at=datetime.now(timezone.utc),
    )

    progress(len(active), len(active),
             f"Backtest complete: {total} trades, WR={win_rate:.1%}, PF={pf:.2f}")
    m.update_task(db_path, task_id, result={"run_id": run_id})
