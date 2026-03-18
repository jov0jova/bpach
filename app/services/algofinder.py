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
  INDICATOR_CATALOG keys are the actual df column names (RSI_14, EMA_50, …).
  Each entry has a semantic "type" that controls exactly how the column is
  used in a condition — Optuna never mixes incompatible units.

  Per-indicator Optuna parameters
  ────────────────────────────────
    use_<col>    0/1 toggle — should this indicator be active this trial?
    thresh_<col> float      — threshold for "lt" and "gt" types only

  Condition types
  ───────────────
    osc         col < threshold     oscillators (RSI, Stoch, MFI, CCI …)
                                    lower value = oversold = bullish entry
    gt          col > threshold     strength/ratio indicators (ADX, volume…)
    sign        col > 0             momentum sign (MACD hist, AO, CMF …)
    flag        col == 1            binary direction flags (Supertrend, PSAR)
    cdl         col > 0             candlestick patterns (store 100, check > 0)
    ma          close > col         price above a moving average (EMA_X)
    band_lower  close < col         price below lower band = oversold bounce
    band_pct    col < threshold     normalised band position (BB %B)
    lt          col < threshold     generic upper-bound filter (NATR, BB wid)

  Cross-conditions — derived automatically, no hardcoded pairs
  ─────────────────────────────────────────────────────────────
  When multiple indicators of the same price-unit type are active,
  CatalogStrategy derives additional conditions at runtime:

    ≥2 active MAs   → EMA_shorter > EMA_longer  (MA alignment / stack)
    band_lower + MA  → band_lower > longest_MA   (floor above trend line)

  This means selecting EMA_20 + EMA_200 automatically implies a golden-cross
  filter — but only when Optuna decides to activate *both*.  No fixed pairs
  are baked in; the relationships emerge from the combination chosen.
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
from ..services.statistics import (
    fdr_correction, permutation_test as _permutation_test,
)
from ..utils.parquet import parquet_path

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Indicator catalogue ───────────────────────────────────────────────────────
# Keys ARE the df column names.  The system infers cross-conditions at runtime
# from the combination of active indicators — no hardcoded pairs here.
# "default": True  →  pre-selected when the user hasn't customised anything.

INDICATOR_CATALOG = OrderedDict([
    # ── Oscillators ──────────────────────────────────────────────────────
    # type "osc": col < threshold  (lower value = oversold = bullish entry)
    ("RSI_7",       {"type": "osc", "range": (15, 50), "label": "RSI(7)",        "cat": "Oscillators"}),
    ("RSI_14",      {"type": "osc", "range": (20, 50), "label": "RSI(14)",       "cat": "Oscillators", "default": True}),
    ("RSI_21",      {"type": "osc", "range": (25, 55), "label": "RSI(21)",       "cat": "Oscillators"}),
    ("STOCH_K",     {"type": "osc", "range": (10, 40), "label": "Stoch %K",      "cat": "Oscillators"}),
    ("STOCHRSI_K",  {"type": "osc", "range": (5,  30), "label": "StochRSI %K",   "cat": "Oscillators"}),
    ("WILLR_14",    {"type": "osc", "range": (-80,-20), "label": "Williams %R",  "cat": "Oscillators"}),
    ("MFI_14",      {"type": "osc", "range": (20, 50), "label": "MFI(14)",       "cat": "Oscillators"}),
    ("CCI_20",      {"type": "osc", "range": (-150,-50),"label": "CCI(20)",      "cat": "Oscillators"}),

    # ── Trend MAs ────────────────────────────────────────────────────────
    # type "ma": close > col
    # Cross-conditions derived automatically at runtime (no hardcoded pairs):
    #   ≥2 active MAs   → shorter_period_EMA > longer_period_EMA  (MA alignment)
    #   band_lower + MA  → band_lower > longest_active_MA          (floor above trend)
    ("EMA_8",       {"type": "ma", "label": "EMA(8)",   "cat": "Trend"}),
    ("EMA_13",      {"type": "ma", "label": "EMA(13)",  "cat": "Trend"}),
    ("EMA_20",      {"type": "ma", "label": "EMA(20)",  "cat": "Trend", "default": True}),
    ("EMA_50",      {"type": "ma", "label": "EMA(50)",  "cat": "Trend", "default": True}),
    ("EMA_100",     {"type": "ma", "label": "EMA(100)", "cat": "Trend"}),
    ("EMA_200",     {"type": "ma", "label": "EMA(200)", "cat": "Trend"}),

    # ── Trend direction flags ─────────────────────────────────────────────
    # type "flag": col == 1  (direction is +1 = bullish)
    ("SUPERT_dir",  {"type": "flag", "label": "Supertrend bullish", "cat": "Trend", "default": True}),
    ("PSAR_dir",    {"type": "flag", "label": "PSAR bullish",       "cat": "Trend"}),
    ("OBV_trend",   {"type": "flag", "label": "OBV trend bullish",  "cat": "Volume"}),

    # ── Trend strength ────────────────────────────────────────────────────
    # type "gt": col > threshold
    ("ADX_14",      {"type": "gt", "range": (15, 35), "label": "ADX(14) strength",  "cat": "Trend", "default": True}),
    ("AROON_up",    {"type": "gt", "range": (50, 90), "label": "Aroon Up strength", "cat": "Trend"}),

    # ── Momentum sign ────────────────────────────────────────────────────
    # type "sign": col > 0  (positive momentum direction)
    ("MACD_hist",   {"type": "sign", "label": "MACD Histogram > 0",    "cat": "Momentum", "default": True}),
    ("AO",          {"type": "sign", "label": "Awesome Oscillator > 0","cat": "Momentum"}),
    ("CMF_20",      {"type": "sign", "label": "Chaikin Money Flow > 0","cat": "Momentum"}),

    # ── Momentum threshold ────────────────────────────────────────────────
    ("ROC_10",      {"type": "gt", "range": (-1, 3), "label": "ROC(10) > threshold %", "cat": "Momentum"}),

    # ── Bands ─────────────────────────────────────────────────────────────
    # type "band_lower": close < col  (oversold below the lower band)
    #   + when any MA is active → col > longest_active_MA  (floor above trend line)
    # type "band_pct": col < threshold  (normalised band position)
    ("KC_lower",    {"type": "band_lower", "label": "Keltner Lower (oversold)",  "cat": "Bands"}),
    ("BB_lower_20", {"type": "band_lower", "label": "BB Lower(20) (oversold)",   "cat": "Bands"}),
    ("BB_pct_20",   {"type": "band_pct", "range": (0.1, 0.4), "label": "BB %B(20) near lower", "cat": "Bands", "default": True}),
    ("BB_width_20", {"type": "gt", "range": (0.005, 0.05), "label": "BB Width(20) > min",       "cat": "Bands"}),

    # ── Volatility ────────────────────────────────────────────────────────
    # type "lt": col < threshold  (avoid high-volatility entries)
    ("NATR_14",     {"type": "lt", "range": (0.5, 4.0), "label": "NATR(14) < max %", "cat": "Volatility"}),

    # ── Volume ────────────────────────────────────────────────────────────
    ("volume_ratio",{"type": "gt", "range": (0.5, 2.5), "label": "Volume ratio > avg", "cat": "Volume", "default": True}),
])

DEFAULT_INDICATORS = [k for k, v in INDICATOR_CATALOG.items() if v.get("default")]


# ── Cross condition catalogue ─────────────────────────────────────────────────
# Keys start with "X_" so they're distinguishable from regular indicator columns.
# These are NEVER stored in the DataFrame — they're computed at runtime via shift().
#
# spec fields:
#   type      : "cross_above" | "cross_below"
#   col_a     : LTF column that does the crossing
#   col_b     : (option 1) LTF column being crossed
#   col_b_htf : (option 2) suffix to find the matching HTF_ column  e.g. "EMA_200"
#   col_b_val : (option 3) scalar level                             e.g. 30 for RSI
#
# All cross conditions fire on exactly ONE bar (the bar where the cross occurs).

CROSS_CATALOG: OrderedDict = OrderedDict([
    # ── EMA pair crosses (same TF) ────────────────────────────────────────
    ("X_ema8_x_ema20",    {"type": "cross_above", "col_a": "EMA_8",   "col_b": "EMA_20",
                           "label": "EMA8 ↑ crosses EMA20",   "cat": "Crosses"}),
    ("X_ema20_x_ema50",   {"type": "cross_above", "col_a": "EMA_20",  "col_b": "EMA_50",
                           "label": "EMA20 ↑ crosses EMA50",  "cat": "Crosses"}),
    ("X_ema50_x_ema200",  {"type": "cross_above", "col_a": "EMA_50",  "col_b": "EMA_200",
                           "label": "EMA50 ↑ crosses EMA200", "cat": "Crosses"}),

    # ── Price crosses a level / band (same TF) ────────────────────────────
    ("X_close_x_ema200",     {"type": "cross_above", "col_a": "close", "col_b": "EMA_200",
                              "label": "Price ↑ crosses EMA200",      "cat": "Crosses"}),
    ("X_close_x_kc_lower",   {"type": "cross_above", "col_a": "close", "col_b": "KC_lower",
                              "label": "Price ↑ crosses Keltner Lower","cat": "Crosses"}),
    ("X_close_x_bb_lower",   {"type": "cross_above", "col_a": "close", "col_b": "BB_lower_20",
                              "label": "Price ↑ crosses BB Lower(20)", "cat": "Crosses"}),

    # ── Oscillator level crosses ──────────────────────────────────────────
    ("X_rsi14_x_30",     {"type": "cross_above", "col_a": "RSI_14",   "col_b_val": 30,
                          "label": "RSI14 ↑ crosses 30 (oversold exit)", "cat": "Crosses"}),
    ("X_rsi14_x_50",     {"type": "cross_above", "col_a": "RSI_14",   "col_b_val": 50,
                          "label": "RSI14 ↑ crosses 50 (momentum)",      "cat": "Crosses"}),
    ("X_stochk_x_20",    {"type": "cross_above", "col_a": "STOCH_K",  "col_b_val": 20,
                          "label": "Stoch %K ↑ crosses 20",              "cat": "Crosses"}),
    ("X_stochrsi_x_20",  {"type": "cross_above", "col_a": "STOCHRSI_K", "col_b_val": 20,
                          "label": "StochRSI ↑ crosses 20",              "cat": "Crosses"}),
    ("X_cci_x_m100",     {"type": "cross_above", "col_a": "CCI_20",   "col_b_val": -100,
                          "label": "CCI ↑ crosses -100 (oversold exit)", "cat": "Crosses"}),
    ("X_willr_x_m50",    {"type": "cross_above", "col_a": "WILLR_14", "col_b_val": -50,
                          "label": "Williams %R ↑ crosses -50",          "cat": "Crosses"}),

    # ── Momentum indicator crosses ─────────────────────────────────────────
    ("X_macd_x_signal",  {"type": "cross_above", "col_a": "MACD",      "col_b": "MACD_signal",
                          "label": "MACD ↑ crosses Signal line",         "cat": "Crosses"}),
    ("X_macd_hist_x_0",  {"type": "cross_above", "col_a": "MACD_hist", "col_b_val": 0,
                          "label": "MACD Histogram ↑ crosses zero",      "cat": "Crosses"}),

    # ── HTF crosses (LTF price / indicator vs HTF level) ──────────────────
    ("X_close_x_htf_ema50",   {"type": "cross_above", "col_a": "close", "col_b_htf": "EMA_50",
                               "label": "Price ↑ crosses HTF EMA50",         "cat": "Crosses/HTF"}),
    ("X_close_x_htf_ema200",  {"type": "cross_above", "col_a": "close", "col_b_htf": "EMA_200",
                               "label": "Price ↑ crosses HTF EMA200",        "cat": "Crosses/HTF"}),
    ("X_close_x_htf_kc_lower",{"type": "cross_above", "col_a": "close", "col_b_htf": "KC_lower",
                               "label": "Price ↑ crosses HTF Keltner Lower", "cat": "Crosses/HTF"}),
    ("X_close_x_htf_bb_lower",{"type": "cross_above", "col_a": "close", "col_b_htf": "BB_lower_20",
                               "label": "Price ↑ crosses HTF BB Lower",      "cat": "Crosses/HTF"}),
    ("X_ema20_x_htf_ema50",   {"type": "cross_above", "col_a": "EMA_20", "col_b_htf": "EMA_50",
                               "label": "LTF EMA20 ↑ crosses HTF EMA50",     "cat": "Crosses/HTF"}),
])


# ── Auto-classification ───────────────────────────────────────────────────────

# Columns that are raw OHLCV, intermediate, or not meaningful standalone
_SKIP_COLS: frozenset = frozenset({
    "open", "high", "low", "close", "volume", "timestamp", "date",
    "entry_signal", "exit_signal", "ema_20_50_cross", "ema_50_200_cross",
    "ema_aligned_bull", "MACD", "MACD_signal", "STOCH_D", "STOCHRSI_D",
    "PSAR_up", "PSAR_down",
    "BB_upper_14", "BB_mid_14", "BB_upper_20", "BB_mid_20",
    "KC_upper", "KC_middle", "DC_upper", "DC_middle",
    "SUPERT_10_3", "SUPERT_14_2", "SUPERT_7_3", "SUPERT_20_2",
    "SUPERT_dir_10_3", "SUPERT_dir_14_2", "SUPERT_dir_7_3", "SUPERT_dir_20_2",
    "ACCB_UPPER", "ACCB_MID",
    "DMP_7", "DMN_7", "DMP_14", "DMN_14", "DMP_21", "DMN_21",
    "ADX_7", "ADX_21", "ATR_7", "ATR_20", "ATR_21", "NATR_7", "NATR_21",
    "candle_body", "candle_range", "upper_wick", "lower_wick",
    "TYPPRICE", "MEDPRICE", "WCLPRICE", "AVGPRICE", "TRANGE",
    "MIDPOINT_14", "MIDPRICE_14", "MAX_14", "MIN_14", "SUM_14",
    "BARS_SINCE_HIGH", "BARS_SINCE_LOW", "PRICE_RANGE_PCT",
    "KST_signal", "PPO", "PPO_signal", "KDJ_D",
    "ICH_tenkan", "ICH_kijun", "ICH_senkou_a", "ICH_senkou_b", "ICH_below_cloud",
    "ALLIGATOR_JAW", "ALLIGATOR_TEETH", "ALLIGATOR_LIPS", "ALLIGATOR_BEAR",
    "AROON_down", "VI_neg", "BEAR_POWER_13", "FRACTAL_BEAR",
    "OBV", "OBV_EMA_12", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "PP", "R1", "R2", "S1", "S2", "PP_dist_pct",
    "OBV_ZSCORE", "ROCR_10", "ROCR100_10", "APO_12_26", "ROCP_10",
    "STDDEV_10", "STDDEV_20", "STDDEV_50", "VAR_10", "VAR_20", "VAR_50",
    "SKEW_20", "KURT_20", "MAD_20", "MEDIAN_20",
    "QUANTILE_75_20", "QUANTILE_25_20", "AUTOCORR_2",
    "BETA_5", "CORREL_CV_14", "CORREL_HL_14", "TSF_14",
    "LINREG_14", "LINREG_INTERCEPT_14", "LINREG_ANGLE_14",
    "TREND_STR_14", "INERTIA_20", "DECAY", "LIN_DECAY_5",
    "AMAT_fast", "SMI_signal", "QQE_LINE", "MACD_cross",
    "HILO_HIGH", "HILO_LOW", "PSL_12", "QSTICK_8", "CFO_9", "CG_10",
    "NVI", "PVI", "PVO_signal", "AR_14", "BR_14", "BRAR",
    "volume_spike", "VOL_RATIO_14",
    "DEMA_ext_9", "DEMA_ext_21", "TEMA_ext_9", "TEMA_ext_21",
    "CDL_SPINNING_TOP", "CDL_DOJI", "CDL_BEAR_MARUBOZU",
    "CDL_HANGING_MAN", "CDL_SHOOTING_STAR", "CDL_GRAVESTONE",
    "CDL_DARK_CLOUD", "CDL_BEAR_ENGULFING", "CDL_BEAR_HARAMI",
    "CDL_BULL_HARAMI", "CDL_TWEEZER_TOP",
    "CDL_3_INSIDE_DOWN", "CDL_3_BLACK_CROWS", "CDL_EVENING_STAR",
    "CDL_OUTSIDE_BAR", "CDL_INSIDE_BAR",
    "HIGH_VOL_REGIME", "BB_SQUEEZE", "LL", "RSI_DIVERG",
    "GMMA_S3", "GMMA_S5", "GMMA_S8", "GMMA_S10", "GMMA_S12", "GMMA_S15",
    "GMMA_L30", "GMMA_L35", "GMMA_L40", "GMMA_L45", "GMMA_L50", "GMMA_L60",
    "HH",
})

_FLAG_COLS: frozenset = frozenset({
    "SUPERT_dir", "PSAR_dir", "OBV_trend", "GMMA_BULL", "ICH_above_cloud",
    "HILO_dir", "PMAX_dir", "TTM_TREND_6", "ALLIGATOR_BULL", "FRACTAL_BULL",
    "CDL_HAMMER", "CDL_INV_HAMMER", "CDL_BULL_ENGULFING", "CDL_MORNING_STAR",
    "CDL_3_WHITE_SOLDIERS", "CDL_PIERCING", "CDL_DRAGONFLY", "CDL_TWEEZER_BOTTOM",
    "CDL_3_INSIDE_UP", "CDL_BULL_MARUBOZU",
})

_SIGN_COLS: frozenset = frozenset({
    "MACD_hist", "AO", "CMF_20", "KVO", "BOP", "PPO_hist", "AROON_osc",
    "KST", "DPO_20", "STC", "FISHER", "SQUEEZE_HIST", "QQE_HIST",
    "BULL_POWER_13", "EMA50_slope", "BIAS_6", "BIAS_14", "BIAS_26",
    "close_vs_EMA_20", "close_vs_EMA_50", "close_vs_EMA_200",
    "CLOSE_VS_VWAP", "LINREG_SLOPE_14", "LINREG_SLOPE_5",
    "ZSCORE_10", "ZSCORE_20", "ZSCORE_50", "KDJ_J", "TSI_13_25",
    "SMI", "COPPOCK", "NET_VOL", "AD", "FI_13", "EOM_14",
    "close_pct_change", "MOM_10", "MOM_20", "PVT",
    "AUTOCORR_1", "PCT_RANK_10", "PCT_RANK_20", "PCT_RANK_50",
    "RVGI_14", "PVO", "TRIX_15",
})

_MA_PREFIXES = (
    "EMA_", "SMA_", "WMA_", "TEMA_", "DEMA_", "HMA_", "ZLEMA_",
    "TRIMA_", "ALMA_", "T3_", "VWMA_",
)
_MA_EXACT: frozenset = frozenset({
    "KAMA", "VIDYA_14", "FWMA_10", "PWMA_10", "SWMA", "HWMA", "MCGD_14", "VWAP",
})


def _auto_cat(col: str) -> str:
    cu = col.upper()
    if any(x in cu for x in ("EMA", "SMA", "WMA", "TEMA", "DEMA", "HMA", "ZLEMA",
                               "TRIMA", "ALMA", "T3_", "KAMA", "VWAP", "VWMA",
                               "VIDYA", "FWMA", "PWMA", "SWMA", "HWMA", "MCGD",
                               "ADX", "AROON", "SUPERT", "PSAR", "GMMA", "HILO",
                               "PMAX", "VI_", "AROONOSC")):
        return "Trend"
    if any(x in cu for x in ("RSI", "STOCH", "MACD", "CCI", "MOM", "ROC", "AO",
                               "DPO", "KST", "TRIX", "WILLR", "UO", "CMO", "CRSI",
                               "KDJ", "TSI", "STC", "FISHER", "SMI", "QQE",
                               "COPPOCK", "PPO", "RVGI")):
        return "Momentum"
    if any(x in cu for x in ("BB_", "KC_", "DC_", "ACCB")):
        return "Bands"
    if any(x in cu for x in ("ATR", "NATR", "HV_", "CHOP", "VHF", "UI_",
                               "SQUEEZE", "VOL_RATIO", "STDDEV", "VAR_",
                               "MASS_INDEX")):
        return "Volatility"
    if any(x in cu for x in ("OBV", "MFI", "CMF", "FI_", "EOM", "AD",
                               "PVT", "NVI", "PVI", "KVO", "VOL", "VOLUME",
                               "NET_VOL", "PVO", "BRAR", "AR_", "BR_")):
        return "Volume"
    if col.startswith("CDL_"):
        return "Patterns"
    if any(x in cu for x in ("ZSCORE", "PCT_RANK", "AUTOCORR", "LINREG",
                               "INERTIA", "BIAS", "CORREL", "TREND_STR")):
        return "Statistical"
    if any(x in cu for x in ("CLOSE_VS", "CLOSE_PCT", "EMA50_SLOPE")):
        return "Price Action"
    return "Other"


def _auto_classify(col: str) -> "dict | None":
    """
    Map any indicator column name to its catalog spec.
    Returns None for columns that should be skipped (OHLCV, derived, etc.).
    """
    if col in _SKIP_COLS:
        return None
    # Skip HTF injected columns — they are handled separately
    if col.startswith("HTF_"):
        return None

    cat = _auto_cat(col)

    # Flags: binary bullish signal
    if col.startswith("CDL_"):
        return {"type": "cdl", "label": col, "cat": "Candlesticks"}
    if col in _FLAG_COLS or col.endswith("_dir") or col.endswith("_BULL"):
        return {"type": "flag", "label": col, "cat": cat}

    # Moving averages (price-scale, close > col)
    for pfx in _MA_PREFIXES:
        if col.startswith(pfx):
            return {"type": "ma", "label": col, "cat": "Trend"}
    if col in _MA_EXACT:
        return {"type": "ma", "label": col, "cat": "Trend"}

    # Sign indicators (col > 0 = bullish momentum direction)
    if col in _SIGN_COLS:
        return {"type": "sign", "label": col, "cat": cat}
    if (col.startswith("BIAS_") or col.startswith("close_vs_")
            or col == "CLOSE_VS_VWAP" or col.startswith("LINREG_SLOPE")
            or col.startswith("ZSCORE_") or col.startswith("PCT_RANK_")
            or col.startswith("ROC_") or col.startswith("MOM_")
            or col.startswith("ROCP_")):
        return {"type": "sign", "label": col, "cat": cat}

    # Oscillators (col < threshold = oversold = bullish)
    if col.startswith("RSI_"):
        return {"type": "osc", "range": (20, 50), "label": col, "cat": "Momentum"}
    if col.startswith("STOCH") or col.startswith("STOCHRSI"):
        return {"type": "osc", "range": (5, 35), "label": col, "cat": "Momentum"}
    if col == "WILLR_14":
        return {"type": "osc", "range": (-80, -20), "label": col, "cat": "Momentum"}
    if col.startswith("MFI_"):
        return {"type": "osc", "range": (20, 50), "label": col, "cat": "Volume"}
    if col.startswith("CCI_"):
        return {"type": "osc", "range": (-150, -50), "label": col, "cat": "Momentum"}
    if col in ("CMO_14", "CRSI", "RSX_14", "UO", "KDJ_K", "CHOP_14"):
        return {"type": "osc", "range": (20, 55), "label": col, "cat": "Momentum"}

    # Band lower (close < col = price at oversold band bottom)
    if col in ("KC_lower", "BB_lower_20", "BB_lower_14", "DC_lower", "ACCB_LOWER"):
        return {"type": "band_lower", "label": col, "cat": "Bands"}
    if "lower" in col.lower() and any(x in col for x in ("BB", "KC", "DC", "ACCB")):
        return {"type": "band_lower", "label": col, "cat": "Bands"}

    # Band pct (col < threshold = near lower band)
    if col.startswith("BB_pct_"):
        return {"type": "band_pct", "range": (0.1, 0.4), "label": col, "cat": "Bands"}

    # Volatility/width upper-limit (col < threshold = not too wide/volatile)
    if col.startswith("NATR_"):
        return {"type": "lt", "range": (0.5, 4.0), "label": col, "cat": "Volatility"}
    if col.startswith("HV_"):
        return {"type": "lt", "range": (10, 60), "label": col, "cat": "Volatility"}
    if col in ("UI_14", "MASS_INDEX"):
        return {"type": "lt", "range": (20, 60), "label": col, "cat": "Volatility"}
    if col.startswith("BB_width_") or col in ("KC_WIDTH", "DC_WIDTH"):
        return {"type": "lt", "range": (0.01, 0.1), "label": col, "cat": "Bands"}

    # Strength (col > threshold = strong trend)
    if col.startswith("ADX_"):
        return {"type": "gt", "range": (15, 35), "label": col, "cat": "Trend"}
    if col == "AROON_up":
        return {"type": "gt", "range": (50, 90), "label": col, "cat": "Trend"}
    if col == "AROONOSC_25":
        return {"type": "gt", "range": (0, 80), "label": col, "cat": "Trend"}
    if col == "volume_ratio":
        return {"type": "gt", "range": (0.5, 2.5), "label": col, "cat": "Volume"}
    if col.startswith("DMP_"):
        return {"type": "gt", "range": (10, 40), "label": col, "cat": "Trend"}
    if col == "VI_pos":
        return {"type": "gt", "range": (0.8, 1.3), "label": col, "cat": "Trend"}
    if col == "VHF_28":
        return {"type": "gt", "range": (0.2, 0.5), "label": col, "cat": "Volatility"}

    return None


# ── Full static column list (all indicators produced by BaseStrategy) ─────────
# This is built once at import time so the selector shows all indicators
# even before any parquet data exists (i.e. before Phase 5 runs).

_ALL_KNOWN_COLS: tuple = (
    # Moving averages
    "EMA_8", "EMA_13", "EMA_20", "EMA_50", "EMA_100", "EMA_200",
    "SMA_8", "SMA_13", "SMA_20", "SMA_21", "SMA_50", "SMA_100", "SMA_200",
    "WMA_8", "WMA_13", "WMA_20", "WMA_21", "WMA_50", "WMA_100", "WMA_200",
    "TEMA_9", "TEMA_21", "TEMA_50",
    "HMA_9", "HMA_20", "HMA_50",
    "ZLEMA_20", "ZLEMA_50",
    "TRIMA_20", "TRIMA_50",
    "ALMA_9", "ALMA_21",
    "T3_5", "T3_10",
    "VIDYA_14", "FWMA_10", "PWMA_10", "SWMA", "HWMA",
    "VWMA_20", "VWMA_50",
    "KAMA", "VWAP", "MCGD_14",
    # Oscillators
    "RSI_7", "RSI_14", "RSI_21",
    "STOCH_K",
    "STOCHRSI_K", "STOCHRSI_K_smooth",
    "WILLR_14",
    "MFI_14",
    "CCI_20",
    "CMO_14", "RSX_14", "CRSI", "UO", "KDJ_K",
    # Momentum sign
    "MACD_hist", "AO", "CMF_20",
    "PPO_hist", "AROON_osc",
    "KST", "DPO_20", "STC", "FISHER",
    "SQUEEZE_HIST", "QQE_HIST",
    "BULL_POWER_13",
    "TSI_13_25", "SMI", "COPPOCK",
    "RVGI_14", "TRIX_15", "PVO",
    "KVO", "FI_13", "EOM_14",
    "AD", "PVT", "NET_VOL", "BOP",
    "KDJ_J",
    # Price-vs-average sign
    "EMA50_slope",
    "close_vs_EMA_20", "close_vs_EMA_50", "close_vs_EMA_200",
    "CLOSE_VS_VWAP",
    "BIAS_6", "BIAS_14", "BIAS_26",
    # Rate of change / momentum
    "ROC_5", "ROC_10", "ROC_14", "ROC_20", "ROC_30",
    "MOM_10", "MOM_20",
    "close_pct_change",
    # Statistical sign
    "LINREG_SLOPE_5", "LINREG_SLOPE_14",
    "ZSCORE_10", "ZSCORE_20", "ZSCORE_50",
    "PCT_RANK_10", "PCT_RANK_20", "PCT_RANK_50",
    "AUTOCORR_1",
    # Trend strength / gt
    "ADX_14",
    "AROON_up", "AROONOSC_25",
    "volume_ratio",
    "DMP_14", "DMP_21",
    "VI_pos", "VHF_28",
    # Flags
    "SUPERT_dir", "PSAR_dir", "OBV_trend",
    "GMMA_BULL", "ICH_above_cloud",
    "HILO_dir", "PMAX_dir",
    "TTM_TREND_6", "ALLIGATOR_BULL", "FRACTAL_BULL",
    # Candlestick flags
    "CDL_HAMMER", "CDL_INV_HAMMER", "CDL_BULL_ENGULFING",
    "CDL_MORNING_STAR", "CDL_3_WHITE_SOLDIERS",
    "CDL_PIERCING", "CDL_DRAGONFLY", "CDL_TWEEZER_BOTTOM",
    "CDL_3_INSIDE_UP", "CDL_BULL_MARUBOZU",
    # Bands lower (oversold bounce)
    "BB_lower_14", "BB_lower_20", "KC_lower", "DC_lower", "ACCB_LOWER",
    # Band pct
    "BB_pct_14", "BB_pct_20",
    # Band / volatility width lt
    "BB_width_14", "BB_width_20", "KC_WIDTH", "DC_WIDTH",
    # Volatility lt
    "NATR_14", "HV_10", "HV_20", "HV_30", "HV_60",
    "UI_14", "MASS_INDEX", "CHOP_14",
)


def _build_full_catalog() -> OrderedDict:
    """Build the complete indicator catalog from all known column names."""
    catalog: OrderedDict = OrderedDict()
    for col in _ALL_KNOWN_COLS:
        if col in INDICATOR_CATALOG:
            catalog[col] = INDICATOR_CATALOG[col]
        else:
            spec = _auto_classify(col)
            if spec is not None:
                catalog[col] = spec
    # Cross conditions are computed at runtime — not stored in parquet — so we
    # always merge them in regardless of what columns exist in the data files.
    catalog.update(CROSS_CATALOG)
    return catalog


# Pre-built at import — always available regardless of parquet state
FULL_INDICATOR_CATALOG: OrderedDict = _build_full_catalog()


def build_dynamic_catalog(parquet_dir: Path, session_id: str,
                           primary_tf: str) -> "OrderedDict":
    """
    Build the indicator catalog from actual parquet column names.
    Reads just the schema (no data) from the first available parquet file.
    Falls back to FULL_INDICATOR_CATALOG (all known columns) when no data exists.
    """
    try:
        import pyarrow.parquet as pq
        session_dir = parquet_dir / session_id
        if not session_dir.exists():
            return FULL_INDICATOR_CATALOG
        # Files are stored as {session_id}/{symbol_dir}/{timeframe}.parquet
        files = sorted(session_dir.glob(f"*/{primary_tf}.parquet"))
        if not files:
            return FULL_INDICATOR_CATALOG
        schema = pq.read_schema(str(files[0]))
        columns = schema.names
    except Exception:
        return FULL_INDICATOR_CATALOG

    catalog: OrderedDict = OrderedDict()
    for col in columns:
        if col in INDICATOR_CATALOG:
            catalog[col] = INDICATOR_CATALOG[col]
        else:
            spec = _auto_classify(col)
            if spec is not None:
                catalog[col] = spec
    # Cross conditions are computed at runtime — not stored in parquet — so
    # always merge them in after scanning the actual file columns.
    catalog.update(CROSS_CATALOG)

    return catalog if catalog else FULL_INDICATOR_CATALOG


# EMA period embedded in column names — used to sort active MAs for cross-conditions
def _ma_period(col: str) -> int:
    """Extract numeric period from EMA_XX column name."""
    try:
        return int(col.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _reconstruct_params(trial_params: dict, selected: list,
                        ic_scores: "dict | None" = None) -> dict:
    """
    Re-derive use_<col> boolean flags from the stored p_use_<col> probabilities.

    Optuna only persists parameters that were passed to trial.suggest_*().
    The use_<col> booleans are computed inside the objective and never suggested,
    so they vanish after the trial.  This function rebuilds them from the stored
    p_use_<col> values using the same threshold formula.
    """
    params = dict(trial_params)
    for col in selected:
        p_key = f"p_use_{col}"
        if p_key in trial_params:
            ic_val   = (ic_scores or {}).get(col, 0.0)
            p_active = 0.3 + min(0.5, ic_val * 5.0)
            params[f"use_{col}"] = int(trial_params[p_key] < p_active)
    return params


def _cross_cond(df: "pd.DataFrame", spec: dict) -> "pd.Series | None":
    """
    Compute a one-bar crossover condition with no lookahead.

    Returns a boolean Series that is True only on the single bar where the
    cross occurs.  Uses shift(1) for the previous bar — never peeks forward.

    spec keys:
      type      : "cross_above" | "cross_below"
      col_a     : LTF column doing the crossing (must exist in df)
      col_b     : (option 1) LTF column being crossed (must exist in df)
      col_b_htf : (option 2) suffix to match an HTF_ column, e.g. "EMA_200"
      col_b_val : (option 3) scalar level, e.g. 30 for RSI
    """
    col_a = spec.get("col_a", "")
    if col_a not in df.columns:
        return None
    a     = df[col_a]
    a_lag = a.shift(1)

    if "col_b" in spec:
        col_b = spec["col_b"]
        if col_b not in df.columns:
            return None
        b     = df[col_b]
        b_lag = b.shift(1)
    elif "col_b_htf" in spec:
        suffix  = spec["col_b_htf"]
        matches = [c for c in df.columns if c.startswith("HTF_") and c.endswith(f"_{suffix}")]
        if not matches:
            return None
        b     = df[matches[0]]
        b_lag = b.shift(1)
    elif "col_b_val" in spec:
        b     = spec["col_b_val"]   # scalar — shift of constant == constant
        b_lag = spec["col_b_val"]
    else:
        return None

    if spec["type"] == "cross_above":
        return (a_lag <= b_lag) & (a > b)
    else:  # cross_below
        return (a_lag >= b_lag) & (a < b)


# ── Execution-TF helpers ──────────────────────────────────────────────────────

class _PassThroughStrategy:
    """
    No-op strategy used when the backtest runs on the execution TF (e.g. 1m).
    entry_signal / exit_signal are already forward-filled from the signal TF
    so we must NOT let _walk_forward_backtest overwrite them.
    """
    params = {}

    def populate_entry_signal(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if "entry_signal" not in df.columns:
            df["entry_signal"] = 0
        return df

    def populate_exit_signal(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if "exit_signal" not in df.columns:
            df["exit_signal"] = 0
        return df

    def run(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return self.populate_exit_signal(self.populate_entry_signal(df))


def _inject_signals_to_exec_tf(signal_df: "pd.DataFrame",
                                exec_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Forward-fill entry/exit signals (and ATR for SL/TP) from signal_df onto
    exec_df bars.  exec_df keeps its own OHLCV — trades execute at exec TF
    prices.  No indicators are re-computed on exec_df.

    Uses merge_asof(direction='backward') so each 1m bar inherits the most
    recent completed signal-TF bar's signals — strictly no lookahead.
    """
    ts_col = "timestamp"
    cols   = [ts_col, "entry_signal", "exit_signal"]
    for atr_col in ("ATR_14", "ATR_20"):           # bring ATR for SL/TP calc
        if atr_col in signal_df.columns:
            cols.append(atr_col)

    sig = signal_df[cols].sort_values(ts_col)
    exc = exec_df.copy()

    # Drop columns that will be injected so merge doesn't create _x/_y suffixes
    exc.drop(columns=[c for c in cols if c != ts_col and c in exc.columns],
             errors="ignore", inplace=True)
    exc = exc.sort_values(ts_col).reset_index(drop=True)

    merged = pd.merge_asof(exc, sig, on=ts_col, direction="backward")
    merged["entry_signal"] = merged["entry_signal"].fillna(0).astype(int)
    merged["exit_signal"]  = merged["exit_signal"].fillna(0).astype(int)
    return merged.reset_index(drop=True)


# ── Strategy class ────────────────────────────────────────────────────────────

# ── Structured strategy slot definitions ─────────────────────────────────────
#
# A strategy is 4 named role-slots that ALL combine with AND:
#
#   TREND   — "What direction is the market?" (state condition)
#   SETUP   — "Is the market set up for entry?" (oscillator / band state)
#   TRIGGER — "What fires the actual entry?" (1-bar crossover / pattern event)
#   CONTEXT — "Does the higher TF confirm?" (optional HTF state condition)
#
# Optuna picks ONE option per slot.  Thresholds are categorical — only
# meaningful round levels (20, 30, 35, 40 … not 66.542).
# Max 4 conditions total.  No OR logic.  No indicator soup.

TREND_SLOT_OPTIONS = [
    "none",             # no trend filter (mean-reversion mode)
    "ema20_gt_ema50",   # EMA(20) > EMA(50)   — short-term bull stack
    "ema50_gt_ema200",  # EMA(50) > EMA(200)  — intermediate bull
    "price_gt_ema50",   # Close > EMA(50)     — above mid-term MA
    "price_gt_ema200",  # Close > EMA(200)    — above long-term MA
    "supertrend_bull",  # Supertrend = Bullish
    "psar_bull",        # Parabolic SAR = Bullish
    "ichimoku_bull",    # Price above Ichimoku cloud
]

SETUP_SLOT_OPTIONS = [
    "none",             # no setup filter
    "rsi_lt_30",        # RSI(14) < 30  — deeply oversold
    "rsi_lt_35",        # RSI(14) < 35  — oversold
    "rsi_lt_40",        # RSI(14) < 40  — mildly oversold
    "rsi_gt_50",        # RSI(14) > 50  — bullish momentum zone
    "stoch_lt_20",      # Stoch %K < 20 — oversold
    "stoch_lt_30",      # Stoch %K < 30
    "macd_positive",    # MACD hist > 0 — momentum positive
    "cci_lt_m100",      # CCI(20) < -100 — oversold
    "mfi_lt_25",        # MFI(14) < 25  — money flow oversold
    "mfi_lt_35",        # MFI(14) < 35
    "bb_below_lower",   # Price below BB Lower(20) — oversold band
    "adx_gt_20",        # ADX(14) > 20  — trending market
    "adx_gt_25",        # ADX(14) > 25  — strong trend
    "vol_spike",        # Volume ratio > 1.5× average
]

TRIGGER_SLOT_OPTIONS = [
    "ema20_cross_ema50",   # EMA(20) crosses above EMA(50)
    "price_cross_ema50",   # Price crosses above EMA(50)
    "price_cross_ema200",  # Price crosses above EMA(200)
    "rsi_cross_30",        # RSI(14) crosses above 30
    "rsi_cross_50",        # RSI(14) crosses above 50
    "stoch_cross_20",      # Stoch %K crosses above 20
    "stochrsi_cross_20",   # StochRSI(K) crosses above 20
    "macd_cross_signal",   # MACD crosses above Signal line
    "macd_hist_cross_0",   # MACD hist crosses above 0
    "cci_cross_m100",      # CCI(20) crosses above -100
    "willr_cross_m50",     # Williams %R crosses above -50
    "hammer_pattern",      # Hammer / Inverted Hammer
    "bull_engulf",         # Bullish Engulfing pattern
    "bb_lower_cross",      # Price crosses above BB Lower(20)
    "supertrend_flip",     # Supertrend flips from Bear → Bull
    "ha_3green",           # 3 consecutive Heikin-Ashi green candles
]

CONTEXT_SLOT_OPTIONS = [
    "none",                # no HTF filter
    "htf_ema50_bull",      # HTF Close > HTF EMA(50)
    "htf_ema200_bull",     # HTF Close > HTF EMA(200)
    "htf_supertrend_bull", # HTF Supertrend = Bullish
    "htf_rsi_not_ob",      # HTF RSI(14) < 70 — not overbought
    "htf_adx_trending",    # HTF ADX(14) > 20 — trend confirmed
]

EXIT_SLOT_OPTIONS = [
    "rsi_gt_70",               # RSI(14) > 70 — overbought
    "rsi_gt_75",               # RSI(14) > 75
    "rsi_gt_80",               # RSI(14) > 80 — strongly overbought
    "ema20_cross_below_ema50", # EMA(20) crosses below EMA(50)
    "supertrend_flip_bear",    # Supertrend flips to Bearish
    "stoch_gt_80",             # Stoch %K > 80 — overbought
    "macd_hist_cross_below_0", # MACD hist crosses below 0
]


def _slot_cross(a: pd.Series, b) -> pd.Series:
    """a crosses above b (scalar or Series).  1-bar event."""
    if isinstance(b, (int, float)):
        return (a > b) & (a.shift(1) <= b)
    return (a > b) & (a.shift(1) <= b.shift(1))


def _slot_cross_below(a: pd.Series, b) -> pd.Series:
    """a crosses below b (scalar or Series).  1-bar event."""
    if isinstance(b, (int, float)):
        return (a < b) & (a.shift(1) >= b)
    return (a < b) & (a.shift(1) >= b.shift(1))


class CatalogStrategy(BaseStrategy):
    """
    Structured strategy: 4 named role-slots all combined with AND.

      TREND   — market direction state filter  (optional)
      SETUP   — oscillator / band state filter (optional)
      TRIGGER — 1-bar crossover or pattern     (required)
      CONTEXT — higher-TF confirmation         (optional)

    Optuna picks ONE option per slot from curated lists.
    Thresholds are categorical round numbers — no float soup.
    Max 4 conditions, always AND.
    """
    name = "CatalogStrategy"

    def __init__(self, params: dict, selected=None):
        super().__init__(params)

    # ── Slot helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _trend(df: pd.DataFrame, slot: str) -> pd.Series:
        c = df.columns
        if slot == "ema20_gt_ema50"   and "EMA_20"  in c and "EMA_50"  in c:
            return df["EMA_20"] > df["EMA_50"]
        if slot == "ema50_gt_ema200"  and "EMA_50"  in c and "EMA_200" in c:
            return df["EMA_50"] > df["EMA_200"]
        if slot == "price_gt_ema50"   and "EMA_50"  in c:
            return df["close"] > df["EMA_50"]
        if slot == "price_gt_ema200"  and "EMA_200" in c:
            return df["close"] > df["EMA_200"]
        if slot == "supertrend_bull"  and "SUPERT_dir" in c:
            return df["SUPERT_dir"] == 1
        if slot == "psar_bull"        and "PSAR_dir" in c:
            return df["PSAR_dir"] == 1
        if slot == "ichimoku_bull"    and "ICH_above_cloud" in c:
            return df["ICH_above_cloud"] == 1
        return pd.Series(True, index=df.index)   # "none" or column missing → pass-through

    @staticmethod
    def _setup(df: pd.DataFrame, slot: str) -> pd.Series:
        c = df.columns
        if slot == "rsi_lt_30"      and "RSI_14"       in c: return df["RSI_14"] < 30
        if slot == "rsi_lt_35"      and "RSI_14"       in c: return df["RSI_14"] < 35
        if slot == "rsi_lt_40"      and "RSI_14"       in c: return df["RSI_14"] < 40
        if slot == "rsi_gt_50"      and "RSI_14"       in c: return df["RSI_14"] > 50
        if slot == "stoch_lt_20"    and "STOCH_K"      in c: return df["STOCH_K"] < 20
        if slot == "stoch_lt_30"    and "STOCH_K"      in c: return df["STOCH_K"] < 30
        if slot == "macd_positive"  and "MACD_hist"    in c: return df["MACD_hist"] > 0
        if slot == "cci_lt_m100"    and "CCI_20"       in c: return df["CCI_20"] < -100
        if slot == "mfi_lt_25"      and "MFI_14"       in c: return df["MFI_14"] < 25
        if slot == "mfi_lt_35"      and "MFI_14"       in c: return df["MFI_14"] < 35
        if slot == "bb_below_lower" and "BB_lower_20"  in c: return df["close"] < df["BB_lower_20"]
        if slot == "adx_gt_20"      and "ADX_14"       in c: return df["ADX_14"] > 20
        if slot == "adx_gt_25"      and "ADX_14"       in c: return df["ADX_14"] > 25
        if slot == "vol_spike"      and "volume_ratio" in c: return df["volume_ratio"] > 1.5
        return pd.Series(True, index=df.index)   # "none" or column missing → pass-through

    @staticmethod
    def _trigger(df: pd.DataFrame, slot: str) -> pd.Series:
        c = df.columns
        if slot == "ema20_cross_ema50"  and "EMA_20" in c and "EMA_50"  in c:
            return _slot_cross(df["EMA_20"], df["EMA_50"])
        if slot == "price_cross_ema50"  and "EMA_50"  in c:
            return _slot_cross(df["close"], df["EMA_50"])
        if slot == "price_cross_ema200" and "EMA_200" in c:
            return _slot_cross(df["close"], df["EMA_200"])
        if slot == "rsi_cross_30"       and "RSI_14"  in c:
            return _slot_cross(df["RSI_14"], 30)
        if slot == "rsi_cross_50"       and "RSI_14"  in c:
            return _slot_cross(df["RSI_14"], 50)
        if slot == "stoch_cross_20"     and "STOCH_K" in c:
            return _slot_cross(df["STOCH_K"], 20)
        if slot == "stochrsi_cross_20"  and "STOCHRSI_K" in c:
            return _slot_cross(df["STOCHRSI_K"], 20)
        if slot == "macd_cross_signal"  and "MACD" in c and "MACD_signal" in c:
            return _slot_cross(df["MACD"], df["MACD_signal"])
        if slot == "macd_hist_cross_0"  and "MACD_hist" in c:
            return _slot_cross(df["MACD_hist"], 0)
        if slot == "cci_cross_m100"     and "CCI_20"  in c:
            return _slot_cross(df["CCI_20"], -100)
        if slot == "willr_cross_m50"    and "WILLR_14" in c:
            return _slot_cross(df["WILLR_14"], -50)
        if slot == "hammer_pattern":
            sig = pd.Series(False, index=df.index)
            for col in ("CDL_HAMMER", "CDL_INV_HAMMER"):
                if col in c:
                    sig |= df[col] > 0
            return sig
        if slot == "bull_engulf" and "CDL_ENGULFING" in c:
            return df["CDL_ENGULFING"] > 0
        if slot == "bb_lower_cross" and "BB_lower_20" in c:
            return _slot_cross(df["close"], df["BB_lower_20"])
        if slot == "supertrend_flip" and "SUPERT_dir" in c:
            return (df["SUPERT_dir"] == 1) & (df["SUPERT_dir"].shift(1) == -1)
        if slot == "ha_3green" and "HA_close" in c and "HA_open" in c:
            green = df["HA_close"] > df["HA_open"]
            return green & green.shift(1).fillna(False) & green.shift(2).fillna(False)
        return pd.Series(False, index=df.index)  # column missing → no signal

    @staticmethod
    def _context(df: pd.DataFrame, slot: str) -> pd.Series:
        c = df.columns
        htf_ema50  = [x for x in c if "HTF_" in x and x.endswith("_EMA_50")]
        htf_ema200 = [x for x in c if "HTF_" in x and x.endswith("_EMA_200")]
        htf_supert = [x for x in c if "HTF_" in x and x.endswith("_SUPERT_dir")]
        htf_rsi    = [x for x in c if "HTF_" in x and x.endswith("_RSI_14")]
        htf_adx    = [x for x in c if "HTF_" in x and x.endswith("_ADX_14")]
        if slot == "htf_ema50_bull"      and htf_ema50:
            return df["close"] > df[htf_ema50[0]]
        if slot == "htf_ema200_bull"     and htf_ema200:
            return df["close"] > df[htf_ema200[0]]
        if slot == "htf_supertrend_bull" and htf_supert:
            return df[htf_supert[0]] == 1
        if slot == "htf_rsi_not_ob"      and htf_rsi:
            return df[htf_rsi[0]] < 70
        if slot == "htf_adx_trending"    and htf_adx:
            return df[htf_adx[0]] > 20
        return pd.Series(True, index=df.index)   # "none" or column missing → pass-through

    # ── Signal population ─────────────────────────────────────────────────

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        trend_cond   = self._trend(df,   p.get("trend_slot",   "none"))
        setup_cond   = self._setup(df,   p.get("setup_slot",   "none"))
        trigger_cond = self._trigger(df, p.get("trigger_slot", "ema20_cross_ema50"))
        context_cond = self._context(df, p.get("context_slot", "none"))
        df["entry_signal"] = (trend_cond & setup_cond & trigger_cond & context_cond).astype(int)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        slot = p.get("exit_slot", "rsi_gt_70")
        c    = df.columns
        cond = pd.Series(False, index=df.index)
        if slot == "rsi_gt_70"               and "RSI_14"   in c: cond = df["RSI_14"] > 70
        elif slot == "rsi_gt_75"             and "RSI_14"   in c: cond = df["RSI_14"] > 75
        elif slot == "rsi_gt_80"             and "RSI_14"   in c: cond = df["RSI_14"] > 80
        elif slot == "ema20_cross_below_ema50" and "EMA_20" in c and "EMA_50" in c:
            cond = _slot_cross_below(df["EMA_20"], df["EMA_50"])
        elif slot == "supertrend_flip_bear"  and "SUPERT_dir" in c:
            cond = (df["SUPERT_dir"] == -1) & (df["SUPERT_dir"].shift(1) == 1)
        elif slot == "stoch_gt_80"           and "STOCH_K"   in c: cond = df["STOCH_K"] > 80
        elif slot == "macd_hist_cross_below_0" and "MACD_hist" in c:
            cond = _slot_cross_below(df["MACD_hist"], 0)
        df["exit_signal"] = cond.astype(int)
        return df


# ── Strategy templates ────────────────────────────────────────────────────────

STRATEGY_TEMPLATES = {
    # All templates use the same slot-based search space.
    # The label is purely informational — Optuna explores all slot combinations.
    "free": {
        "label": "Free Search",
        "description": "Full slot search: Optuna picks trend, setup, trigger and exit freely.",
        "icon": "bi-shuffle",
    },
    "trend_follow": {
        "label": "Trend Following",
        "description": "Trend filter + momentum setup + crossover trigger. Buy the trend.",
        "icon": "bi-arrow-up-right",
    },
    "mean_revert": {
        "label": "Mean Reversion",
        "description": "Oversold setup + bounce trigger. RSI / Stoch / BB extremes.",
        "icon": "bi-arrow-left-right",
    },
    "breakout": {
        "label": "Breakout",
        "description": "Price or MA crossover trigger with ADX / volume confirmation.",
        "icon": "bi-graph-up",
    },
    "momentum": {
        "label": "Momentum",
        "description": "MACD / RSI momentum setup + crossover trigger.",
        "icon": "bi-lightning-fill",
    },
    "scalp": {
        "label": "Scalp / Short-term",
        "description": "Fast oscillator crossovers and candle patterns for short holds.",
        "icon": "bi-clock-history",
    },
}


# ── Regime detection ──────────────────────────────────────────────────────────

def _detect_regime(df: "pd.DataFrame") -> str:
    """Classify the dominant market regime: trending_bull|trending_bear|ranging|volatile."""
    if len(df) < 50:
        return "all"
    try:
        if "EMA50_slope" in df.columns:
            slope = float(df["EMA50_slope"].dropna().median())
        elif "EMA_50" in df.columns:
            slope = float(df["EMA_50"].pct_change(5).dropna().median()) * 100
        else:
            slope = 0.0
        natr = 0.0
        if "NATR_14" in df.columns:
            natr = float(df["NATR_14"].dropna().median())
        elif "ATR_14" in df.columns and "close" in df.columns:
            natr = float((df["ATR_14"] / df["close"]).dropna().median()) * 100
        if natr > 4.0:
            return "volatile"
        if slope > 0.05:
            return "trending_bull"
        if slope < -0.05:
            return "trending_bear"
        return "ranging"
    except Exception:
        return "all"


# ── Scoring helper ─────────────────────────────────────────────────────────────

def _score_wfo_results(all_wfo: list, min_oos_trades: int = 5) -> float:
    """
    Composite score:
      Sharpe (main) + log(1+return)*0.1 + (capped_PF-1)*0.2 + pair_coverage_bonus

    Rewards strategies that generalize across pairs.
    Returns -999 if insufficient trades or no valid Sharpe (too few trades to
    compute return variance — guards against single-trade PF explosions).
    """
    if not all_wfo:
        return -999.0
    avg_trades = float(np.mean([w["oos_trades"] for w in all_wfo]))
    if avg_trades < min_oos_trades:
        return -999.0

    sharpes = [w["oos_sharpe"] for w in all_wfo]
    returns = [w["oos_return"] for w in all_wfo]
    pfs     = [w.get("oos_profit_factor", 1.0) for w in all_wfo]

    # Drop NaN sharpes (arise when std(returns) == 0, i.e. too few / identical trades).
    # If no pair produced a valid Sharpe the result is meaningless — reject it.
    valid_sharpes = [s for s in sharpes if s is not None and not np.isnan(s)]
    if not valid_sharpes:
        return -999.0

    # Cap PF at 5 to prevent strategies with 0 losing OOS trades from
    # dominating the score with PF values in the millions.
    pf_capped = [min(p, 5.0) for p in pfs if p is not None and not np.isnan(p)]
    pf_score  = (np.mean(pf_capped) - 1.0) * 0.2 if pf_capped else 0.0

    pairs_pos = sum(1 for r in returns if r > 0)
    coverage  = pairs_pos / len(all_wfo)
    cov_bonus = 0.3 if coverage >= 0.5 else 0.0

    return float(
        np.mean(valid_sharpes)
        + np.log1p(max(0, np.mean(returns))) * 0.1
        + pf_score
        + cov_bonus
    )


def _run_wfo_for_trial(df: "pd.DataFrame", strategy, config: dict,
                       params: dict,
                       exec_df: "pd.DataFrame | None" = None) -> dict:
    """
    Run fast WFO for a single df using the trial's TP/SL/trailing params.

    If exec_df is provided (execution TF, e.g. 1m):
      - Generate entry/exit signals on df (signal TF)
      - Forward-fill those signals + ATR onto exec_df via merge_asof
      - Backtest on exec_df using real 1m OHLCV prices
      - _PassThroughStrategy ensures WFO doesn't re-run signals on each split
    """
    from .backtest import _walk_forward_backtest as _wfbt, _simple_backtest

    # Compute signals on signal TF
    enriched = df.copy(deep=False)
    enriched = strategy.populate_entry_signal(enriched)
    enriched = strategy.populate_exit_signal(enriched)

    if exec_df is not None and len(exec_df) >= 50:
        # Inject signals into execution TF; backtest runs on 1m OHLCV
        run_df      = _inject_signals_to_exec_tf(enriched, exec_df)
        run_strategy = _PassThroughStrategy()
    else:
        run_df       = enriched
        run_strategy = strategy

    bt_kw = dict(
        initial_capital = config.get("initial_capital", 10_000),
        fee_rate        = config.get("fee_rate", 0.001),
        slippage        = config.get("slippage", 0.0005),
        position_size   = config.get("position_size", 0.1),
        sl_mode=params.get("sl_mode","none"),
        sl_pct=params.get("sl_pct", 0.02), sl_atr_period=14,
        sl_atr_multiplier=params.get("sl_atr_multiplier", 2.0),
        tp_mode=params.get("tp_mode","none"),
        tp_pct=params.get("tp_pct", 0.04), tp_atr_period=14,
        tp_atr_multiplier=params.get("tp_atr_multiplier", 4.0),
        tp_rr_ratio=params.get("tp_rr_ratio", 2.0),
        trail_mode=params.get("trail_mode","none"),
        trail_pct=params.get("trail_pct", 0.02), trail_atr_period=14,
        trail_atr_multiplier=params.get("trail_atr_multiplier", 1.5),
    )
    return _wfbt(run_df, run_strategy,
                 n_splits=config.get("wfo_splits", 3),
                 train_ratio=config.get("wfo_train_ratio", 0.7),
                 bt_kwargs=bt_kw, fast_mode=True)


# ── Optuna objective ──────────────────────────────────────────────────────────

def _sample_slot_params(trial, has_htf: bool = False) -> dict:
    """
    Sample all strategy slot choices + risk-management params for one Optuna trial.
    Called by both single-objective and multi-objective functions.
    """
    params = {}
    # ── Strategy structure (4 role-slots, ONE choice each, always AND) ───────
    params["trend_slot"]   = trial.suggest_categorical("trend_slot",   TREND_SLOT_OPTIONS)
    params["setup_slot"]   = trial.suggest_categorical("setup_slot",   SETUP_SLOT_OPTIONS)
    params["trigger_slot"] = trial.suggest_categorical("trigger_slot", TRIGGER_SLOT_OPTIONS)
    params["context_slot"] = trial.suggest_categorical(
        "context_slot",
        CONTEXT_SLOT_OPTIONS if has_htf else ["none"],
    )
    params["exit_slot"]    = trial.suggest_categorical("exit_slot",    EXIT_SLOT_OPTIONS)

    # ── Risk management (SL / TP / Trailing — still optimised per trial) ─────
    params["sl_mode"]              = trial.suggest_categorical("sl_mode",   ["none", "fixed", "atr"])
    params["sl_pct"]               = trial.suggest_float("sl_pct",          0.01, 0.08)
    params["sl_atr_multiplier"]    = trial.suggest_float("sl_atr_multiplier", 1.0, 4.0)
    params["tp_mode"]              = trial.suggest_categorical("tp_mode",   ["none", "fixed", "atr", "rr"])
    params["tp_pct"]               = trial.suggest_float("tp_pct",          0.02, 0.15)
    params["tp_atr_multiplier"]    = trial.suggest_float("tp_atr_multiplier", 2.0, 8.0)
    params["tp_rr_ratio"]          = trial.suggest_float("tp_rr_ratio",      1.0, 4.0)
    params["trail_mode"]           = trial.suggest_categorical("trail_mode", ["none", "fixed", "atr"])
    params["trail_pct"]            = trial.suggest_float("trail_pct",        0.005, 0.05)
    params["trail_atr_multiplier"] = trial.suggest_float("trail_atr_multiplier", 0.5, 3.0)
    return params


def _objective(trial, dfs: list, config: dict,
               selected: list = None, has_htf: bool = False,
               ic_scores: dict = None,
               min_oos_trades: int = 5) -> float:
    """
    Single-objective Optuna search using structured role-slots.

    Optuna picks ONE option per slot (trend / setup / trigger / context / exit)
    from curated lists.  All conditions combine with AND.  No OR soup.
    Composite score: Sharpe + log(return) + (capped PF - 1) + pair coverage bonus.
    """
    params   = _sample_slot_params(trial, has_htf=has_htf)
    strategy = CatalogStrategy(params)
    all_wfo  = []
    for signal_df, exec_df in dfs:
        if len(signal_df) < 100:
            continue
        try:
            all_wfo.append(_run_wfo_for_trial(signal_df, strategy, config, params,
                                              exec_df=exec_df))
        except Exception:
            continue
    return _score_wfo_results(all_wfo, min_oos_trades)


def _objective_multi(trial, dfs: list, config: dict,
                     selected: list = None, has_htf: bool = False,
                     ic_scores: dict = None) -> tuple:
    """
    Multi-objective: maximise (oos_sharpe, oos_return) simultaneously.
    Returns Pareto-optimal strategies.
    """
    params   = _sample_slot_params(trial, has_htf=has_htf)
    strategy = CatalogStrategy(params)
    all_wfo  = []
    for signal_df, exec_df in dfs:
        if len(signal_df) < 100:
            continue
        try:
            all_wfo.append(_run_wfo_for_trial(signal_df, strategy, config, params,
                                              exec_df=exec_df))
        except Exception:
            continue

    if not all_wfo:
        return (-999.0, -999.0)
    avg_trades = float(np.mean([w["oos_trades"] for w in all_wfo]))
    if avg_trades < 5:
        return (-999.0, -999.0)
    sharpes = [w["oos_sharpe"] for w in all_wfo]
    valid_sharpes = [s for s in sharpes if s is not None and not np.isnan(s)]
    if not valid_sharpes:
        return (-999.0, -999.0)
    return (
        float(np.mean(valid_sharpes)),
        float(np.mean([w["oos_return"] for w in all_wfo])),
    )

# ── Main task ─────────────────────────────────────────────────────────────────

def _extract_ic_scores(config: dict) -> dict:
    """
    Extract {col: max_abs_ic} from IC analysis results stored in config.
    Used to bias indicator activation probability.
    """
    ic_result = config.get("ic_result") or {}
    top_overall = ic_result.get("top_overall") or []
    return {row["col"]: row.get("max_abs_ic", 0.0) for row in top_overall if row.get("col")}


def run_algofinder(task_id: str, db_path: Path, session_id: str,
                   parquet_dir: Path, timeframes: list,
                   n_trials: int = 100, config: dict = None, stop_event=None) -> None:
    """
    Background task: smart Optuna search for the most profitable strategy.

    New in v2:
    - IC-guided indicator activation (uses IC analysis results)
    - Strategy templates (bias search toward proven archetypes)
    - Multi-objective NSGA-II support
    - Dynamic TP/SL/trailing also optimised per trial
    - Cross-pair generalization scoring
    - Composite score: Sharpe + log(return) + PF + coverage bonus

    config keys:
        selected_indicators: list of indicator keys
        selected_pairs:      list of symbols (subset)
        template:            'free' | 'trend_follow' | 'mean_revert' | 'breakout' |
                             'momentum' | 'scalp'
        multi_objective:     bool (use NSGA-II Pareto search)
        ic_result:           dict from IC analysis (used to bias sampling)
        min_oos_trades:      int (minimum OOS trades to consider valid)
    """
    if config is None:
        config = {}

    # Template selection — label only, slot search is unconstrained
    template_key = config.get("template", "free")
    template     = STRATEGY_TEMPLATES.get(template_key, STRATEGY_TEMPLATES["free"])
    # `selected` kept for IC analysis compatibility; not used for strategy generation
    selected = config.get("selected_indicators") or DEFAULT_INDICATORS
    selected = [k for k in selected if k in FULL_INDICATOR_CATALOG]
    if not selected:
        selected = DEFAULT_INDICATORS

    # IC scores for guided sampling
    ic_scores = _extract_ic_scores(config)

    multi_obj     = bool(config.get("multi_objective", False))
    min_oos_trades = int(config.get("min_oos_trades", 5))

    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    progress(0, n_trials, "Loading data for Algo Finder…")

    pairs  = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]

    selected_pairs = config.get("selected_pairs") or []
    if selected_pairs:
        active = [p for p in active if p["symbol"] in selected_pairs]
    if not active:
        active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]

    primary_tf   = timeframes[0] if timeframes else "1h"
    higher_tfs   = timeframes[1:] if len(timeframes) > 1 else []
    execution_tf = config.get("execution_tf")          # None = same as signal TF
    if execution_tf == primary_tf:
        execution_tf = None                            # no-op when TFs are equal

    # dfs stores (signal_df, exec_df_or_None) tuples
    dfs            = []
    htf_found_flag = [False]
    load_lock      = threading.Lock()

    def _load_pair(pair: dict):
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

        # Load execution TF (e.g. 1m) — raw OHLCV only, no indicator computation
        exec_df = None
        if execution_tf:
            exec_path = parquet_path(parquet_dir, session_id, symbol, execution_tf)
            if exec_path.exists():
                try:
                    exec_df = pd.read_parquet(exec_path)
                    if len(exec_df) < 50:
                        exec_df = None
                except Exception as e:
                    logger.warning("AlgoFinder exec TF load %s %s: %s",
                                   symbol, execution_tf, e)

        return (df, exec_df)

    sample_pairs = active[:config.get("max_pairs", 30)]
    workers = min((os.cpu_count() or 4) * 4, len(sample_pairs))
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

    # Detect dominant regime across pairs for regime-aware reporting
    regimes = [_detect_regime(sig_df) for sig_df, _ in dfs[:10]]
    dominant_regime = max(set(regimes), key=regimes.count) if regimes else "all"

    ic_guided = bool(ic_scores)
    tf_desc   = primary_tf + (f" + HTF: {', '.join(higher_tfs)}" if htf_found else "")
    mode_desc = ("NSGA-II multi-obj" if multi_obj
                 else f"TPE {'IC-guided' if ic_guided else 'free'}")
    progress(0, n_trials,
             f"{n_trials} trials on {len(dfs)} pairs [{tf_desc}] | "
             f"Template={template['label']} | Sampler={mode_desc} | "
             f"Regime={dominant_regime}")

    # ── Create Optuna study (persistent across runs) ─────────────────────────
    optuna_db = db_path.parent / "optuna_studies.db"
    storage = optuna.storages.RDBStorage(f"sqlite:///{optuna_db}")
    study_name = f"{session_id}__{template_key}__{'multi' if multi_obj else 'single'}"
    if multi_obj:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        )
    prior_trials = len(study.trials)
    if prior_trials:
        progress(0, n_trials,
                 f"Resuming study '{study_name}' — {prior_trials} trials already done")

    trial_count  = [0]
    counter_lock = threading.Lock()

    def callback(study, trial):
        with counter_lock:
            trial_count[0] += 1
            count = trial_count[0]
        if count % 5 == 0:
            if multi_obj:
                best_str = f"{len(study.best_trials)} Pareto solutions"
            else:
                try:
                    best_str = f"best={study.best_value:.3f}"
                except Exception:
                    best_str = "searching…"
            progress(count, n_trials,
                     f"Trial {count}/{n_trials} — {best_str}")

    n_jobs = min(config.get("n_jobs", 2), (os.cpu_count() or 1) * 2)
    if multi_obj:
        study.optimize(
            lambda t: _objective_multi(t, dfs, config, selected,
                                       has_htf=htf_found, ic_scores=ic_scores),
            n_trials=n_trials, n_jobs=n_jobs,
            callbacks=[callback], show_progress_bar=False,
        )
        # Collect Pareto front trials
        pareto = [t for t in study.best_trials]
        # Sort by sharpe + return combined
        pareto.sort(
            key=lambda t: (t.values[0] + t.values[1] * 0.01 if t.values else -999),
            reverse=True,
        )
        top_trials = pareto
    else:
        study.optimize(
            lambda t: _objective(t, dfs, config, selected,
                                 has_htf=htf_found, ic_scores=ic_scores,
                                 min_oos_trades=min_oos_trades),
            n_trials=n_trials, n_jobs=n_jobs,
            callbacks=[callback], show_progress_bar=False,
        )
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and (t.value or -999) > -999]
        completed.sort(key=lambda t: (t.value or -999), reverse=True)
        top_trials = completed

    any_profitable = any(
        (t.values[0] if multi_obj else t.value or 0) > 0
        for t in top_trials
    )

    # Clear old results for this session before saving new ones
    m.clear_algo_results(db_path, session_id)

    # ── Per-strategy evaluation + permutation test ────────────────────────────
    # We run full WFO on top strategies, collect all OOS trades, then compute
    # a permutation-based p-value per strategy.
    # After collecting p-values for all strategies we apply BH FDR correction.
    strategy_metrics = []   # list of metric dicts, one per strategy
    strategy_trades  = []   # list of trade lists (OOS), one per strategy

    n_top = len(top_trials)
    progress(n_trials, n_trials + n_top, "Evaluating top strategies + significance tests…")

    for rank, trial in enumerate(top_trials, 1):
        # With slot-based params all values are persisted via trial.suggest_*
        params   = dict(trial.params)
        strategy = CatalogStrategy(params)

        all_wfo  = []
        oos_trades_this = []   # all OOS trades from every pair × fold

        for signal_df, exec_df in dfs:
            try:
                wfo_r = _run_wfo_for_trial(signal_df, strategy, config, params,
                                           exec_df=exec_df)
                all_wfo.append(wfo_r)
                # Collect OOS trade pnls for permutation test (top 3 only)
                if rank <= 3:
                    try:
                        bt_kw = dict(
                            initial_capital=config.get("initial_capital", 10_000),
                            fee_rate=config.get("fee_rate", 0.001),
                            slippage=config.get("slippage", 0.0005),
                            position_size=config.get("position_size", 0.1),
                            fast_mode=False,
                        )
                        enriched = strategy.run(signal_df.copy())
                        if exec_df is not None and len(exec_df) >= 50:
                            enriched = _inject_signals_to_exec_tf(enriched, exec_df)
                        if "entry_signal" in enriched.columns:
                            bt_r = _simple_backtest(enriched, **bt_kw)
                            oos_trades_this.extend(bt_r.get("trades_list", []))
                    except Exception:
                        pass
            except Exception:
                continue

        def _avg(key):
            vals = [w[key] for w in all_wfo if w.get(key) is not None]
            return float(np.mean(vals)) if vals else 0.0

        pairs_positive = sum(1 for w in all_wfo if w.get("oos_return", 0) > 0)
        pair_coverage  = pairs_positive / len(all_wfo) if all_wfo else 0.0

        # Permutation test (only if enough trades)
        perm_result = {}
        if len(oos_trades_this) >= 10:
            try:
                perm_result = _permutation_test(oos_trades_this, n_perms=500)
            except Exception as e:
                logger.warning("Permutation test failed for strategy %d: %s", rank, e)

        p_val = perm_result.get("p_value", 1.0)

        strategy_metrics.append({
            "rank":          rank,
            "trial":         trial,
            "params":        params,
            "is_return":     _avg("is_return"),
            "oos_return":    _avg("oos_return"),
            "win_rate":      _avg("oos_win_rate"),
            "sharpe":        _avg("oos_sharpe"),
            "max_drawdown":  _avg("oos_max_dd"),
            "profit_factor": _avg("oos_profit_factor"),
            "calmar_ratio":  _avg("oos_calmar"),
            "sortino_ratio": _avg("oos_sortino"),
            "expectancy":    _avg("oos_expectancy"),
            "pair_coverage": pair_coverage,
            "p_value":       p_val,
        })
        strategy_trades.append(oos_trades_this)

        progress(n_trials + rank, n_trials + n_top,
                 f"Evaluated strategy {rank}/{n_top} | p={p_val:.3f}")

    # ── BH FDR correction across all strategies ────────────────────────────────
    raw_p_values = [s["p_value"] for s in strategy_metrics]
    fdr_result   = fdr_correction(raw_p_values, alpha=0.05)
    adj_p_values = fdr_result.get("adjusted_p_values", raw_p_values)
    is_sig_list  = fdr_result.get("is_significant", [False] * len(raw_p_values))
    n_sig        = fdr_result.get("n_significant", 0)

    # ── Save results ───────────────────────────────────────────────────────────
    for i, sm in enumerate(strategy_metrics):
        rank   = sm["rank"]
        params = sm["params"]
        rules  = _describe_rules(params, selected)
        adj_p  = adj_p_values[i] if i < len(adj_p_values) else 1.0
        is_sig = bool(is_sig_list[i]) if i < len(is_sig_list) else False

        m.save_algo_result(
            db_path, session_id, None, rank,
            f"{template['label']}_#{rank}",
            params, rules,
            {
                "is_return":       sm["is_return"],
                "oos_return":      sm["oos_return"],
                "win_rate":        sm["win_rate"],
                "sharpe":          sm["sharpe"],
                "max_drawdown":    sm["max_drawdown"],
                "profit_factor":   sm["profit_factor"],
                "calmar_ratio":    sm["calmar_ratio"],
                "sortino_ratio":   sm["sortino_ratio"],
                "expectancy":      sm["expectancy"],
                "pair_coverage":   sm["pair_coverage"],
                "regime":          dominant_regime,
                "template":        template_key,
                "p_value":         sm["p_value"],
                "adjusted_p":      adj_p,
                "is_significant":  is_sig,
                "n_trials_tested": n_trials,
            },
        )

    if not top_trials:
        summary = (f"Algo Finder finished — no valid strategies found. "
                   f"Try more trials, different template, or add more data.")
    elif any_profitable:
        n = len(top_trials)
        best_s = (top_trials[0].values[0] if multi_obj else top_trials[0].value) or 0
        summary = (f"✓ {n} strategies found | Regime={dominant_regime} | "
                   f"Template={template['label']} | Best score={best_s:.3f} | "
                   f"FDR significant: {n_sig}/{n}")
    else:
        summary = (f"Algo Finder done — {len(top_trials)} strategies saved but none profitable OOS. "
                   "Try more trials or a different template.")

    progress(n_trials + n_top, n_trials + n_top, summary)
    m.update_task(db_path, task_id, result={
        "top_count":        len(top_trials),
        "any_profitable":   any_profitable,
        "regime":           dominant_regime,
        "template":         template_key,
        "ic_guided":        ic_guided,
        "multi_objective":  multi_obj,
        "n_significant":    n_sig,
        "fdr_alpha":        0.05,
    })


# ── Rule description ──────────────────────────────────────────────────────────

_TREND_LABELS = {
    "none":             "none — no trend filter",
    "ema20_gt_ema50":   "EMA(20) > EMA(50)",
    "ema50_gt_ema200":  "EMA(50) > EMA(200)",
    "price_gt_ema50":   "Price > EMA(50)",
    "price_gt_ema200":  "Price > EMA(200)",
    "supertrend_bull":  "Supertrend = Bullish",
    "psar_bull":        "Parabolic SAR = Bullish",
    "ichimoku_bull":    "Price above Ichimoku cloud",
}
_SETUP_LABELS = {
    "none":             "none — no setup filter",
    "rsi_lt_30":        "RSI(14) < 30  — deeply oversold",
    "rsi_lt_35":        "RSI(14) < 35  — oversold",
    "rsi_lt_40":        "RSI(14) < 40  — mildly oversold",
    "rsi_gt_50":        "RSI(14) > 50  — bullish momentum zone",
    "stoch_lt_20":      "Stoch %K < 20 — oversold",
    "stoch_lt_30":      "Stoch %K < 30",
    "macd_positive":    "MACD Histogram > 0",
    "cci_lt_m100":      "CCI(20) < -100 — oversold",
    "mfi_lt_25":        "MFI(14) < 25  — money flow oversold",
    "mfi_lt_35":        "MFI(14) < 35",
    "bb_below_lower":   "Price below BB Lower(20) — oversold band",
    "adx_gt_20":        "ADX(14) > 20  — trending market",
    "adx_gt_25":        "ADX(14) > 25  — strong trend",
    "vol_spike":        "Volume ratio > 1.5× average",
}
_TRIGGER_LABELS = {
    "ema20_cross_ema50":   "EMA(20) crosses above EMA(50)",
    "price_cross_ema50":   "Price crosses above EMA(50)",
    "price_cross_ema200":  "Price crosses above EMA(200)",
    "rsi_cross_30":        "RSI(14) crosses above 30 — exits oversold",
    "rsi_cross_50":        "RSI(14) crosses above 50 — momentum flip",
    "stoch_cross_20":      "Stoch %K crosses above 20",
    "stochrsi_cross_20":   "StochRSI(K) crosses above 20",
    "macd_cross_signal":   "MACD crosses above Signal line",
    "macd_hist_cross_0":   "MACD Histogram crosses above 0",
    "cci_cross_m100":      "CCI(20) crosses above -100",
    "willr_cross_m50":     "Williams %R crosses above -50",
    "hammer_pattern":      "Hammer / Inverted Hammer candle",
    "bull_engulf":         "Bullish Engulfing pattern",
    "bb_lower_cross":      "Price crosses above BB Lower(20)",
    "supertrend_flip":     "Supertrend flips Bullish",
    "ha_3green":           "3 consecutive Heikin-Ashi green candles",
}
_CONTEXT_LABELS = {
    "none":                "none — no HTF filter",
    "htf_ema50_bull":      "[HTF] Price > HTF EMA(50)",
    "htf_ema200_bull":     "[HTF] Price > HTF EMA(200)",
    "htf_supertrend_bull": "[HTF] Supertrend = Bullish",
    "htf_rsi_not_ob":      "[HTF] RSI(14) < 70 — not overbought",
    "htf_adx_trending":    "[HTF] ADX(14) > 20 — trend confirmed",
}
_EXIT_LABELS = {
    "rsi_gt_70":               "RSI(14) > 70 — overbought",
    "rsi_gt_75":               "RSI(14) > 75",
    "rsi_gt_80":               "RSI(14) > 80 — strongly overbought",
    "ema20_cross_below_ema50": "EMA(20) crosses below EMA(50)",
    "supertrend_flip_bear":    "Supertrend flips Bearish",
    "stoch_gt_80":             "Stoch %K > 80 — overbought",
    "macd_hist_cross_below_0": "MACD Histogram crosses below 0",
}


def _describe_sl_tp(params: dict) -> str:
    """One-line SL/TP/trail summary for the rules block."""
    parts = []
    sl = params.get("sl_mode", "none")
    if sl == "fixed":
        parts.append(f"SL {params.get('sl_pct', 0.02) * 100:.1f}%")
    elif sl == "atr":
        parts.append(f"SL {params.get('sl_atr_multiplier', 2.0):.1f}×ATR")
    tp = params.get("tp_mode", "none")
    if tp == "fixed":
        parts.append(f"TP {params.get('tp_pct', 0.04) * 100:.1f}%")
    elif tp == "atr":
        parts.append(f"TP {params.get('tp_atr_multiplier', 4.0):.1f}×ATR")
    elif tp == "rr":
        parts.append(f"TP {params.get('tp_rr_ratio', 2.0):.1f}:1 R:R")
    trail = params.get("trail_mode", "none")
    if trail == "fixed":
        parts.append(f"Trail {params.get('trail_pct', 0.02) * 100:.1f}%")
    elif trail == "atr":
        parts.append(f"Trail {params.get('trail_atr_multiplier', 1.5):.1f}×ATR")
    return "  " + " | ".join(parts) if parts else "  none"


def _describe_rules(params: dict, selected=None) -> str:
    """
    Human-readable description of a slot-based strategy.

    Output example:
        ENTRY  (all conditions AND, entry on trigger bar):
          [TREND]    EMA(20) > EMA(50)
          [SETUP]    RSI(14) < 35  — oversold
          [TRIGGER]  MACD crosses above Signal line  ←
          [CONTEXT]  [HTF] Price > HTF EMA(50)

        EXIT:
          RSI(14) > 75

        RISK:
          SL 2.0×ATR | TP 3.0:1 R:R
    """
    trend   = params.get("trend_slot",   "none")
    setup   = params.get("setup_slot",   "none")
    trigger = params.get("trigger_slot", "ema20_cross_ema50")
    context = params.get("context_slot", "none")
    exit_s  = params.get("exit_slot",    "rsi_gt_70")

    lines = ["ENTRY  (all conditions AND, entry fires on trigger bar):"]
    if trend != "none":
        lines.append(f"  [TREND]    {_TREND_LABELS.get(trend, trend)}")
    if setup != "none":
        lines.append(f"  [SETUP]    {_SETUP_LABELS.get(setup, setup)}")
    lines.append(    f"  [TRIGGER]  {_TRIGGER_LABELS.get(trigger, trigger)}  ←")
    if context != "none":
        lines.append(f"  [CONTEXT]  {_CONTEXT_LABELS.get(context, context)}")

    lines += [
        "",
        "EXIT:",
        f"  {_EXIT_LABELS.get(exit_s, exit_s)}",
    ]
    sl_line = _describe_sl_tp(params)
    if sl_line.strip() != "none":
        lines += ["", "RISK:", sl_line]
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

    for signal_df, exec_df in dfs:
        if len(signal_df) < 100:
            continue
        try:
            all_wfo.append(_run_wfo_for_trial(signal_df, strategy, config, params,
                                              exec_df=exec_df))
        except Exception:
            continue

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
    primary_tf   = timeframes[0] if timeframes else "1h"
    higher_tfs   = timeframes[1:] if len(timeframes) > 1 else []
    execution_tf = config.get("execution_tf")
    if execution_tf == primary_tf:
        execution_tf = None

    # dfs stores (signal_df, exec_df_or_None) tuples
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
        exec_df = None
        if execution_tf:
            exec_path = parquet_path(parquet_dir, session_id, symbol, execution_tf)
            if exec_path.exists():
                try:
                    exec_df = pd.read_parquet(exec_path)
                    if len(exec_df) < 50:
                        exec_df = None
                except Exception:
                    pass
        return (df, exec_df)

    workers_a = min((os.cpu_count() or 4) * 4, len(active[:20]))
    with ThreadPoolExecutor(max_workers=workers_a) as exe:
        for result in exe.map(_load_pair_a, active[:20]):
            if result is not None:
                dfs.append(result)

    if not dfs:
        m.update_task(db_path, task_id, status="error",
                      error="No data available.", message="Download and add indicators first.")
        return

    progress(0, n_trials, f"Running {n_trials} Path A trials on {len(dfs)} pairs…")

    optuna_db_a = db_path.parent / "optuna_studies.db"
    storage_a = optuna.storages.RDBStorage(f"sqlite:///{optuna_db_a}")
    import hashlib as _hashlib
    _entry_hash = _hashlib.md5(condition_code.encode()).hexdigest()[:8]
    study_name_a = f"{session_id}__path_a__{_entry_hash}"
    study = optuna.create_study(
        study_name=study_name_a,
        storage=storage_a,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    prior_trials_a = len(study.trials)
    if prior_trials_a:
        progress(0, n_trials,
                 f"Resuming Path A study — {prior_trials_a} trials already done")
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

    n_jobs_a = (os.cpu_count() or 1) * 2
    study.optimize(
        lambda trial: _path_a_objective(
            trial, dfs, config, condition_code, indicator_filters),
        n_trials=n_trials,
        n_jobs=n_jobs_a,
        callbacks=[callback],
        show_progress_bar=False,
    )

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value > -999]
    completed.sort(key=lambda t: t.value, reverse=True)
    top5 = completed[:5]
    any_profitable_a = any(t.value > 0 for t in top5)

    for rank, trial in enumerate(top5, 1):
        params   = trial.params
        strategy = PathAStrategy(params, condition_code, indicator_filters)

        all_wfo = []
        for signal_df, exec_df in dfs:
            try:
                all_wfo.append(_run_wfo_for_trial(signal_df, strategy, config, params,
                                                  exec_df=exec_df))
            except Exception:
                continue

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

    if not top5:
        pa_summary = "Path A finished — no valid strategies (all trials had < 5 trades)."
    elif any_profitable_a:
        pa_summary = f"Path A complete — {len(top5)} strategies found, best score: {top5[0].value:.3f}"
    else:
        pa_summary = f"Path A finished — {len(top5)} strategies saved but none profitable OOS."
    progress(n_trials, n_trials, pa_summary)
    m.update_task(db_path, task_id,
                  result={"top_count": len(top5), "mode": "path_a",
                          "any_profitable": any_profitable_a})


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
