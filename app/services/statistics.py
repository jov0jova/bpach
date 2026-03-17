"""
Statistical validation module for quant backtesting.

Implements:
  • Monte Carlo simulation — resample trade returns to show outcome distribution
  • Bootstrap confidence intervals — CI on Sharpe, Win Rate, Max Drawdown
  • Permutation test — p-value: is strategy return distinguishable from noise?
  • Benjamini-Hochberg FDR correction — correct for multiple strategy comparisons
  • Benchmark comparison — buy-and-hold return / alpha
  • Portfolio-level metrics — combined equity curve across pairs

Why each matters
────────────────
  Monte Carlo: your backtest shows ONE path through history. MC shows the range
    of plausible outcomes if trade order had been different. The 5th-percentile
    equity curve is your realistic worst case.

  Bootstrap CI: a Sharpe of 1.2 on 30 trades has a wide CI. You need to know
    whether the CI brackets 0 before trusting the result.

  Permutation test: if we randomly shuffle entry signals, what fraction of runs
    beat our strategy? If >5% do, the strategy isn't significant.

  FDR correction: running 1000 Optuna trials means testing 1000 hypotheses.
    Without correction, we expect 50 false positives at α=0.05. BH FDR limits
    the expected proportion of false discoveries among accepted strategies.

  Benchmark: a strategy that returns 40% when BTC went up 150% is actually
    underperforming. Alpha = strategy return - benchmark return.

  Portfolio: 20 pairs that are 95% correlated give no diversification. Portfolio
    Sharpe measures the actual risk-adjusted return you'd get deploying capital.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_simulation(
    trades: list[dict],
    n_sims: int = 2000,
    initial_capital: float = 10_000,
    ci_levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    seed: int = 42,
) -> dict:
    """
    Resample trade returns with replacement to produce a distribution of outcomes.

    Parameters
    ----------
    trades       : list of trade dicts with 'pnl_pct' and 'position_size' keys
    n_sims       : number of Monte Carlo paths
    initial_capital : starting capital for each simulated path
    ci_levels    : percentiles to report (0–1 fractions)
    seed         : random seed for reproducibility

    Returns
    -------
    dict with:
        sharpe_p5/p50/p95       — Sharpe ratio percentiles
        maxdd_p5/p50/p95        — Max drawdown percentiles (worst = p95 for DD)
        total_return_p5/p50/p95 — Total return percentiles
        win_rate_p5/p50/p95     — Win rate percentiles
        equity_p5/p50/p95       — List of equity curves at each percentile
        prob_ruin               — Fraction of paths where capital drops > 50%
        prob_profitable         — Fraction of paths with positive total return
        n_trades                — Number of trades in original sample
    """
    if not trades:
        return _empty_mc()

    rng = np.random.default_rng(seed)
    # Convert trades to return array: r_i = pnl_pct / 100 * position_size
    returns = np.array([t["pnl_pct"] / 100 * t.get("position_size", 0.1)
                        for t in trades], dtype=float)
    n = len(returns)

    all_sharpes   = np.empty(n_sims)
    all_maxdds    = np.empty(n_sims)
    all_tot_ret   = np.empty(n_sims)
    all_win_rates = np.empty(n_sims)
    # Store equity curves for a sample of simulations for chart rendering
    # Only store every 10th to save memory
    stored_equity = []

    for s in range(n_sims):
        sample  = rng.choice(returns, size=n, replace=True)
        eq      = _equity_from_returns(sample, initial_capital)
        sr      = _sharpe(sample)
        dd      = _max_drawdown_pct(eq)
        tot_ret = (eq[-1] / initial_capital - 1) * 100
        wr      = float(np.mean(sample > 0))

        all_sharpes[s]   = sr
        all_maxdds[s]    = dd
        all_tot_ret[s]   = tot_ret
        all_win_rates[s] = wr
        if s % (n_sims // 100) == 0:          # keep ~100 equity paths
            stored_equity.append(eq.tolist())

    def _pct(arr, p):
        return float(np.percentile(arr, p * 100))

    result = {}
    for metric, arr in [
        ("sharpe",       all_sharpes),
        ("total_return", all_tot_ret),
        ("win_rate",     all_win_rates),
    ]:
        for p in ci_levels:
            result[f"{metric}_p{int(p*100)}"] = _pct(arr, p)

    # For drawdown, high percentile = worst case (sign convention: positive number = loss)
    for p in ci_levels:
        result[f"maxdd_p{int(p*100)}"] = _pct(all_maxdds, p)

    result["prob_ruin"]        = float(np.mean(all_tot_ret < -50))
    result["prob_profitable"]  = float(np.mean(all_tot_ret > 0))
    result["n_trades"]         = n
    result["n_sims"]           = n_sims
    result["equity_paths"]     = stored_equity   # sample of paths for chart

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    trades: list[dict],
    n_boot: int = 1000,
    ci: float = 0.95,
    initial_capital: float = 10_000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals for key metrics.

    Returns lower/upper bounds for Sharpe, Win Rate, Max Drawdown,
    Profit Factor, and Total Return.
    """
    if not trades:
        return {}

    rng     = np.random.default_rng(seed)
    returns = np.array([t["pnl_pct"] / 100 * t.get("position_size", 0.1)
                        for t in trades], dtype=float)
    pnls    = np.array([t["pnl_pct"] for t in trades], dtype=float)
    n       = len(returns)
    alpha   = (1 - ci) / 2

    sharpes    = np.empty(n_boot)
    winrates   = np.empty(n_boot)
    maxdds     = np.empty(n_boot)
    tot_rets   = np.empty(n_boot)
    pfs        = np.empty(n_boot)

    for b in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        r_boot  = returns[idx]
        p_boot  = pnls[idx]

        sharpes[b]  = _sharpe(r_boot)
        winrates[b] = float(np.mean(r_boot > 0))
        eq          = _equity_from_returns(r_boot, initial_capital)
        maxdds[b]   = _max_drawdown_pct(eq)
        tot_rets[b] = (eq[-1] / initial_capital - 1) * 100
        gains  = p_boot[p_boot > 0].sum()
        losses = abs(p_boot[p_boot < 0].sum())
        pfs[b] = gains / losses if losses > 0 else (2.0 if gains > 0 else 0.0)

    def _ci(arr):
        return (float(np.percentile(arr, alpha * 100)),
                float(np.percentile(arr, (1 - alpha) * 100)))

    lo_s, hi_s   = _ci(sharpes)
    lo_w, hi_w   = _ci(winrates)
    lo_d, hi_d   = _ci(maxdds)
    lo_r, hi_r   = _ci(tot_rets)
    lo_p, hi_p   = _ci(pfs)

    return {
        "ci_level":             ci,
        "sharpe_lo":            lo_s, "sharpe_hi": hi_s,
        "win_rate_lo":          lo_w, "win_rate_hi": hi_w,
        "max_drawdown_lo":      lo_d, "max_drawdown_hi": hi_d,
        "total_return_lo":      lo_r, "total_return_hi": hi_r,
        "profit_factor_lo":     lo_p, "profit_factor_hi": hi_p,
        "n_trades":             n,
        "n_boot":               n_boot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Permutation test (strategy significance)
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(
    trades: list[dict],
    n_perms: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Permutation test: what fraction of random orderings of entry signals
    produce a Sharpe >= the observed Sharpe?

    This is equivalent to: "is the mean trade return distinguishable from 0
    via randomisation?"

    Returns
    -------
    dict with:
        p_value            — fraction of permutations beating observed Sharpe
        observed_sharpe    — actual strategy Sharpe
        perm_sharpe_mean   — mean Sharpe across permutations (≈ 0 expected)
        is_significant     — p_value < 0.05
    """
    if len(trades) < 10:
        return {"p_value": 1.0, "observed_sharpe": 0.0,
                "perm_sharpe_mean": 0.0, "is_significant": False}

    rng     = np.random.default_rng(seed)
    returns = np.array([t["pnl_pct"] / 100 * t.get("position_size", 0.1)
                        for t in trades], dtype=float)
    obs_sr  = _sharpe(returns)

    perm_srs = np.empty(n_perms)
    for p in range(n_perms):
        shuffled      = rng.permutation(returns)
        perm_srs[p]   = _sharpe(shuffled)

    p_value = float(np.mean(perm_srs >= obs_sr))
    return {
        "p_value":          p_value,
        "observed_sharpe":  float(obs_sr),
        "perm_sharpe_mean": float(np.mean(perm_srs)),
        "perm_sharpe_std":  float(np.std(perm_srs)),
        "is_significant":   p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benjamini-Hochberg FDR correction
# ─────────────────────────────────────────────────────────────────────────────

def fdr_correction(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg procedure for controlling False Discovery Rate.

    When testing m strategies, we expect α*m false positives at raw significance.
    BH guarantees that the expected proportion of false discoveries among
    *accepted* strategies ≤ α.

    Parameters
    ----------
    p_values : list of raw p-values (one per strategy)
    alpha    : FDR level (default 0.05)

    Returns
    -------
    dict with:
        adjusted_p_values  — BH-adjusted p-values (same order as input)
        is_significant     — boolean mask at FDR level α
        n_significant      — number of strategies passing FDR threshold
        n_tested           — total strategies tested
        fdr_threshold      — the actual p-value threshold used
    """
    p_arr = np.array(p_values, dtype=float)
    m     = len(p_arr)
    if m == 0:
        return {
            "adjusted_p_values": [], "is_significant": [],
            "n_significant": 0, "n_tested": 0, "fdr_threshold": 0.0,
        }

    # Sort p-values and track original order
    order    = np.argsort(p_arr)
    p_sorted = p_arr[order]
    ranks    = np.arange(1, m + 1)

    # BH critical values: (i/m) * α
    thresholds = (ranks / m) * alpha

    # Find the largest k where p_(k) ≤ (k/m)*α
    below_threshold = p_sorted <= thresholds
    if below_threshold.any():
        k_max = int(np.where(below_threshold)[0].max())
        fdr_threshold = float(thresholds[k_max])
    else:
        k_max         = -1
        fdr_threshold = 0.0

    # Adjusted p-values: p_adj_(i) = min over j≥i of (m/j)*p_(j)
    # (Benjamini & Hochberg step-up procedure)
    adj_sorted = np.minimum.accumulate((m / ranks * p_sorted)[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)

    # Map back to original order
    adj_p  = np.empty(m)
    adj_p[order] = adj_sorted
    sig    = adj_p <= alpha

    return {
        "adjusted_p_values": adj_p.tolist(),
        "is_significant":    sig.tolist(),
        "n_significant":     int(sig.sum()),
        "n_tested":          m,
        "fdr_threshold":     fdr_threshold,
        "alpha":             alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark comparison
# ─────────────────────────────────────────────────────────────────────────────

def compute_benchmark(
    df: "pd.DataFrame",
    initial_capital: float = 10_000,
    column: str = "close",
) -> dict:
    """
    Compute buy-and-hold return for the same data period as the backtest.

    Parameters
    ----------
    df      : OHLCV DataFrame with 'close' column
    column  : price column to use for buy-and-hold

    Returns
    -------
    dict with:
        benchmark_return    — total return %
        benchmark_sharpe    — annualized Sharpe of daily returns
        benchmark_maxdd     — max drawdown %
        benchmark_start     — first close price
        benchmark_end       — last close price
    """
    if df is None or column not in df.columns or len(df) < 2:
        return {
            "benchmark_return": 0.0, "benchmark_sharpe": 0.0,
            "benchmark_maxdd": 0.0,  "benchmark_start": 0.0,
            "benchmark_end": 0.0,
        }

    prices  = df[column].values.astype(float)
    prices  = prices[np.isfinite(prices)]
    if len(prices) < 2:
        return {
            "benchmark_return": 0.0, "benchmark_sharpe": 0.0,
            "benchmark_maxdd": 0.0,  "benchmark_start": 0.0,
            "benchmark_end": 0.0,
        }

    tot_ret = (prices[-1] / prices[0] - 1) * 100
    rets    = np.diff(prices) / prices[:-1]
    sr      = _sharpe(rets)
    eq      = prices / prices[0] * initial_capital
    dd      = _max_drawdown_pct(eq)

    return {
        "benchmark_return": float(tot_ret),
        "benchmark_sharpe": float(sr),
        "benchmark_maxdd":  float(dd),
        "benchmark_start":  float(prices[0]),
        "benchmark_end":    float(prices[-1]),
    }


def compute_alpha(strategy_return: float, benchmark_return: float) -> float:
    """Alpha = strategy return minus benchmark return (simple, not CAPM)."""
    return strategy_return - benchmark_return


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_metrics(
    pair_trade_lists: dict[str, list[dict]],
    initial_capital: float = 10_000,
    allocation: str = "equal",
) -> dict:
    """
    Combine per-pair trade lists into a portfolio-level equity curve and metrics.

    The portfolio equity curve is constructed by:
    1. Allocating equal capital to each pair (allocation='equal')
    2. Running each pair's trades against its allocated capital slice
    3. Summing equity curves aligned on a common datetime index

    Parameters
    ----------
    pair_trade_lists : {symbol: [trades]} — each trade has entry_time, exit_time, pnl_pct, position_size
    initial_capital  : total capital
    allocation       : 'equal' (only mode supported currently)

    Returns
    -------
    dict with portfolio_sharpe, portfolio_maxdd, portfolio_return,
         portfolio_sortino, pair_correlation_mean, pair_equity_curves (for chart),
         correlation_matrix
    """
    if not pair_trade_lists:
        return _empty_portfolio()

    n_pairs = len(pair_trade_lists)
    if n_pairs == 0:
        return _empty_portfolio()

    per_pair_capital = initial_capital / n_pairs

    # Build daily-return series per pair from trade list
    pair_returns = {}
    for symbol, trades in pair_trade_lists.items():
        if not trades:
            continue
        rets = np.array([t["pnl_pct"] / 100 * t.get("position_size", 0.1)
                         for t in trades], dtype=float)
        pair_returns[symbol] = rets

    if not pair_returns:
        return _empty_portfolio()

    # For portfolio-level Sharpe we need aligned return series
    # Simplest: combine all trade returns across pairs, weighted by per-pair allocation
    combined_returns = np.concatenate(list(pair_returns.values()))
    port_sharpe      = _sharpe(combined_returns)

    # Downside for Sortino
    down             = combined_returns[combined_returns < 0]
    d_std            = float(np.std(down)) + 1e-9 if len(down) > 0 else 1e-9
    mean_r           = float(np.mean(combined_returns))
    port_sortino     = float(mean_r / d_std * np.sqrt(252))

    # Equity curves per pair
    pair_equities = {}
    for symbol, rets in pair_returns.items():
        eq = _equity_from_returns(rets, per_pair_capital)
        pair_equities[symbol] = eq.tolist()

    # Portfolio equity: sum of per-pair equity curves at each step
    # Pad shorter curves with their last value
    max_len = max(len(v) for v in pair_equities.values())
    combined_eq = np.zeros(max_len)
    for eq_list in pair_equities.values():
        arr = np.array(eq_list, dtype=float)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), mode="edge")
        combined_eq += arr

    port_maxdd    = _max_drawdown_pct(combined_eq)
    port_tot_ret  = (combined_eq[-1] / initial_capital - 1) * 100

    # Pair return correlation (using per-trade returns padded to same length)
    corr_mean = _mean_pair_correlation(pair_returns)

    return {
        "portfolio_sharpe":       float(port_sharpe),
        "portfolio_sortino":      float(port_sortino),
        "portfolio_maxdd":        float(port_maxdd),
        "portfolio_return":       float(port_tot_ret),
        "pair_correlation_mean":  float(corr_mean),
        "n_pairs":                n_pairs,
        "pair_equity_curves":     pair_equities,
        "portfolio_equity":       combined_eq.tolist(),
    }


def _mean_pair_correlation(pair_returns: dict[str, np.ndarray]) -> float:
    """Estimate mean pairwise Spearman rank correlation between pair return series."""
    symbols = list(pair_returns.keys())
    if len(symbols) < 2:
        return 0.0

    # Pad all series to the same length with zeros
    max_len = max(len(v) for v in pair_returns.values())
    mat = np.zeros((max_len, len(symbols)))
    for j, sym in enumerate(symbols):
        arr = pair_returns[sym]
        mat[:len(arr), j] = arr

    from scipy.stats import spearmanr
    try:
        if mat.shape[1] < 2:
            return 0.0
        corr_mat, _ = spearmanr(mat)
        if corr_mat.ndim == 0:     # only 2 pairs
            return float(corr_mat)
        # Take mean of upper triangle (excluding diagonal)
        n = corr_mat.shape[0]
        vals = [corr_mat[i, j] for i in range(n) for j in range(i+1, n)]
        return float(np.mean(vals)) if vals else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _equity_from_returns(returns: np.ndarray, initial_capital: float) -> np.ndarray:
    """Compound equity curve from per-trade fractional returns."""
    factors = 1.0 + returns
    # Use cumprod for compounding
    cum     = np.cumprod(factors)
    return initial_capital * np.concatenate([[1.0], cum])


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (assumes returns are fractional, not %)."""
    if len(returns) < 2:
        return 0.0
    m   = float(np.mean(returns))
    s   = float(np.std(returns)) + 1e-9
    return float(m / s * np.sqrt(periods_per_year))


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Maximum drawdown as a positive percentage."""
    if len(equity) < 2:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dds   = (peaks - equity) / (peaks + 1e-12) * 100
    return float(np.max(dds))


def _empty_mc() -> dict:
    zero_pcts = {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    result = {}
    for metric in ["sharpe", "total_return", "win_rate", "maxdd"]:
        for pct in [5, 25, 50, 75, 95]:
            result[f"{metric}_p{pct}"] = 0.0
    result.update({"prob_ruin": 0.0, "prob_profitable": 0.0,
                   "n_trades": 0, "n_sims": 0, "equity_paths": []})
    return result


def _empty_portfolio() -> dict:
    return {
        "portfolio_sharpe": 0.0, "portfolio_sortino": 0.0,
        "portfolio_maxdd": 0.0,  "portfolio_return": 0.0,
        "pair_correlation_mean": 0.0, "n_pairs": 0,
        "pair_equity_curves": {}, "portfolio_equity": [],
    }
