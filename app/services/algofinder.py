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


# ── Strategy class ────────────────────────────────────────────────────────────

class CatalogStrategy(BaseStrategy):
    """
    Strategy built dynamically from a user-selected subset of INDICATOR_CATALOG.

    Optuna controls per-indicator:
      • use_<col>    0/1 — activate this indicator for this trial
      • thresh_<col> float — threshold (osc / gt / lt / band_pct types only)

    Cross-conditions are derived automatically — no hardcoded pairs:
      ≥2 active MAs    → EMA_shorter > EMA_longer   (MA alignment)
      band_lower + MA  → band_lower  > longest_MA   (floor above trend line)
    """
    name = "CatalogStrategy"

    def __init__(self, params: dict, selected: list):
        super().__init__(params)
        self._selected = selected

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        # trigger_logic controls how oversold/cross/cdl conditions combine:
        #   "and" → ALL active triggers must fire  (default, strict)
        #   "or"  → ANY active trigger is enough   (more signals, less overfitting)
        trigger_logic = p.get("trigger_logic", "and")

        # Two accumulator series: triggers (osc/band/cdl/cross) and filters (ma/trend/htf)
        trigger_cond = pd.Series(False, index=df.index)  # OR-built
        filter_cond  = pd.Series(True,  index=df.index)  # AND-built
        n_triggers   = 0

        active_mas   = []   # (period_int, col) for active "ma" entries
        active_bands = []   # col names for active "band_lower" entries

        for col in self._selected:
            if not p.get(f"use_{col}", 0):
                continue
            spec = FULL_INDICATOR_CATALOG.get(col)
            if spec is None:
                continue

            kind = spec["type"]

            # ── Dynamic crossover conditions (triggers, no precomputed column) ──
            if kind in ("cross_above", "cross_below"):
                cross = _cross_cond(df, spec)
                if cross is not None:
                    trigger_cond |= cross
                    n_triggers   += 1
                continue

            # Regular conditions require the column to exist in df
            if col not in df.columns:
                continue

            # ── Trigger conditions (oversold / pattern — combine with OR or AND) ──
            if kind in ("osc", "band_pct"):
                c = df[col] < p[f"thresh_{col}"]
                trigger_cond |= c
                n_triggers   += 1

            elif kind == "band_lower":
                c = df["close"] < df[col]
                trigger_cond |= c
                n_triggers   += 1
                active_bands.append(col)

            elif kind == "cdl":
                trigger_cond |= (df[col] > 0)
                n_triggers   += 1

            # ── Filter conditions (context/trend — always AND) ──────────────
            elif kind == "lt":
                filter_cond &= df[col] < p[f"thresh_{col}"]

            elif kind == "gt":
                filter_cond &= df[col] > p[f"thresh_{col}"]

            elif kind == "sign":
                filter_cond &= df[col] > 0

            elif kind == "flag":
                filter_cond &= df[col] == 1

            elif kind == "ma":
                filter_cond &= df["close"] > df[col]
                active_mas.append((_ma_period(col), col))

        # ── Combine trigger group with filter group ───────────────────────────
        if n_triggers == 0:
            # No triggers selected — fall back to pure filter logic
            cond = filter_cond
        elif trigger_logic == "or":
            # ANY trigger fires → AND with all filters
            cond = trigger_cond & filter_cond
        else:
            # ALL triggers must fire (AND mode) → AND with filters
            # Rebuild trigger_cond as AND
            cond = filter_cond
            for col in self._selected:
                if not p.get(f"use_{col}", 0):
                    continue
                spec = FULL_INDICATOR_CATALOG.get(col)
                if spec is None:
                    continue
                kind = spec["type"]
                if kind in ("cross_above", "cross_below"):
                    cross = _cross_cond(df, spec)
                    if cross is not None:
                        cond &= cross
                elif col not in df.columns:
                    continue
                elif kind in ("osc", "band_pct"):
                    cond &= df[col] < p[f"thresh_{col}"]
                elif kind == "band_lower":
                    cond &= df["close"] < df[col]
                elif kind == "cdl":
                    cond &= df[col] > 0

        # ── Derived cross-conditions (price-unit indicators) ──────────────
        # MA alignment: if ≥2 MAs active, require shorter-period > longer-period
        active_mas.sort()                       # ascending = fastest first
        if len(active_mas) >= 2:
            fast_col = active_mas[0][1]         # shortest period (fastest MA)
            slow_col = active_mas[-1][1]        # longest period  (slowest MA)
            if fast_col in df.columns and slow_col in df.columns:
                cond &= df[fast_col] > df[slow_col]

        # Band-vs-MA: if any lower-band + any MA are active,
        # require band_lower > longest_MA (floor is above the trend line)
        if active_bands and active_mas:
            anchor_ma = active_mas[-1][1]       # most conservative = longest period
            if anchor_ma in df.columns:
                for band_col in active_bands:
                    if band_col in df.columns:
                        cond &= df[band_col] > df[anchor_ma]

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

        htf_ema200_cols = [c for c in df.columns if "HTF_" in c and c.endswith("_EMA_200")]
        if p.get("use_htf_ema200", 0) and htf_ema200_cols:
            cond &= df["close"] > df[htf_ema200_cols[0]]

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


# ── Strategy templates ────────────────────────────────────────────────────────

STRATEGY_TEMPLATES = {
    "free": {
        "label": "Free Search",
        "description": "Unconstrained search across all selected indicators.",
        "indicators": None,
        "icon": "bi-shuffle",
    },
    "trend_follow": {
        "label": "Trend Following",
        "description": "EMA alignment + momentum confirmation. Buy the trend.",
        "indicators": ["EMA_20", "EMA_50", "EMA_200", "ADX_14", "MACD_hist",
                       "SUPERT_dir", "RSI_14", "volume_ratio"],
        "icon": "bi-arrow-up-right",
    },
    "mean_revert": {
        "label": "Mean Reversion",
        "description": "Oversold bounce from lower band. RSI/Stoch oversold + support.",
        "indicators": ["RSI_14", "STOCHRSI_K", "BB_pct_20", "BB_lower_20",
                       "MFI_14", "CCI_20", "WILLR_14", "CMF_20"],
        "icon": "bi-arrow-left-right",
    },
    "breakout": {
        "label": "Breakout",
        "description": "Price breaking out with volume confirmation and momentum.",
        "indicators": ["BB_width_20", "volume_ratio", "ADX_14", "MACD_hist",
                       "ROC_10", "AO", "SUPERT_dir", "EMA_50"],
        "icon": "bi-graph-up",
    },
    "momentum": {
        "label": "Momentum",
        "description": "Strong momentum with trend and volume confirmation.",
        "indicators": ["RSI_7", "MACD_hist", "AO", "ROC_10", "CMF_20",
                       "EMA_20", "volume_ratio", "ADX_14"],
        "icon": "bi-lightning-fill",
    },
    "scalp": {
        "label": "Scalp / Short-term",
        "description": "Fast oscillators for short-term high-frequency entries.",
        "indicators": ["RSI_7", "STOCHRSI_K", "BB_pct_20", "MACD_hist",
                       "NATR_14", "volume_ratio", "EMA_8", "EMA_20"],
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
                       params: dict) -> dict:
    """Run fast WFO for a single df using the trial's TP/SL/trailing params."""
    from .backtest import _walk_forward_backtest as _wfbt, _simple_backtest
    enriched = df.copy(deep=False)
    enriched = strategy.populate_entry_signal(enriched)
    enriched = strategy.populate_exit_signal(enriched)
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
    return _wfbt(enriched, strategy,
                 n_splits=config.get("wfo_splits", 3),
                 train_ratio=config.get("wfo_train_ratio", 0.7),
                 bt_kwargs=bt_kw, fast_mode=True)


# ── Optuna objective ──────────────────────────────────────────────────────────

def _objective(trial, dfs: list, config: dict,
               selected: list, has_htf: bool = False,
               ic_scores: dict = None,
               min_oos_trades: int = 5) -> float:
    """
    Smart single-objective Optuna search.

    Improvements vs. v1:
    - IC-biased indicator activation: high-IC indicators more likely to be active.
    - Dynamic TP/SL/trailing also optimised per trial.
    - Cross-pair generalization reward.
    - Composite score: Sharpe + log(return) + (PF-1) + coverage bonus.
    """
    params = {}
    params["rsi_exit_period"]  = trial.suggest_categorical("rsi_exit_period", [7, 14, 21])
    params["rsi_exit_min"]     = trial.suggest_float("rsi_exit_min", 60, 85)
    params["trigger_logic"]    = trial.suggest_categorical("trigger_logic", ["and", "or"])

    # Dynamic stop/profit params (optimised alongside entry rules)
    params["sl_mode"]              = trial.suggest_categorical("sl_mode", ["none","fixed","atr"])
    params["sl_pct"]               = trial.suggest_float("sl_pct", 0.01, 0.08)
    params["sl_atr_multiplier"]    = trial.suggest_float("sl_atr_multiplier", 1.0, 4.0)
    params["tp_mode"]              = trial.suggest_categorical("tp_mode", ["none","fixed","atr","rr"])
    params["tp_pct"]               = trial.suggest_float("tp_pct", 0.02, 0.15)
    params["tp_atr_multiplier"]    = trial.suggest_float("tp_atr_multiplier", 2.0, 8.0)
    params["tp_rr_ratio"]          = trial.suggest_float("tp_rr_ratio", 1.0, 4.0)
    params["trail_mode"]           = trial.suggest_categorical("trail_mode", ["none","fixed","atr"])
    params["trail_pct"]            = trial.suggest_float("trail_pct", 0.005, 0.05)
    params["trail_atr_multiplier"] = trial.suggest_float("trail_atr_multiplier", 0.5, 3.0)

    # Per-indicator parameters (IC-biased activation)
    for col in selected:
        spec = FULL_INDICATOR_CATALOG.get(col)
        if spec is None:
            continue
        # IC-guided: high |IC| → higher activation probability
        ic_val = (ic_scores or {}).get(col, 0.0)
        p_active = 0.3 + min(0.5, ic_val * 5.0)
        raw = trial.suggest_float(f"p_use_{col}", 0.0, 1.0)
        params[f"use_{col}"] = int(raw < p_active)
        if spec["type"] in ("osc", "gt", "lt", "band_pct") and "range" in spec:
            lo, hi = spec["range"]
            params[f"thresh_{col}"] = trial.suggest_float(f"thresh_{col}", lo, hi)

    if has_htf:
        params["use_htf_trend"]      = trial.suggest_categorical("use_htf_trend", [0, 1])
        params["use_htf_supertrend"] = trial.suggest_categorical("use_htf_supertrend", [0, 1])
        params["use_htf_rsi_filter"] = trial.suggest_categorical("use_htf_rsi_filter", [0, 1])
        params["htf_rsi_max"]        = trial.suggest_float("htf_rsi_max", 30, 70)
        params["use_htf_adx_filter"] = trial.suggest_categorical("use_htf_adx_filter", [0, 1])
        params["htf_adx_min"]        = trial.suggest_float("htf_adx_min", 15, 40)
        params["use_htf_ema200"]     = trial.suggest_categorical("use_htf_ema200", [0, 1])

    # Require at least 1 active LTF condition — HTF-only strategies fire on every
    # bar where the HTF context is satisfied and produce meaningless statistics.
    n_ltf_active = sum(1 for col in selected if params.get(f"use_{col}", 0))
    if n_ltf_active == 0:
        return -999.0

    strategy = CatalogStrategy(params, selected)
    all_wfo  = []
    for df in dfs:
        if len(df) < 100:
            continue
        try:
            all_wfo.append(_run_wfo_for_trial(df, strategy, config, params))
        except Exception:
            continue

    return _score_wfo_results(all_wfo, min_oos_trades)


def _objective_multi(trial, dfs: list, config: dict,
                     selected: list, has_htf: bool = False,
                     ic_scores: dict = None) -> tuple:
    """
    Multi-objective: maximise (oos_sharpe, oos_return) simultaneously.
    Returns Pareto-optimal strategies.
    """
    params = {}
    params["rsi_exit_period"]  = trial.suggest_categorical("rsi_exit_period", [7, 14, 21])
    params["rsi_exit_min"]     = trial.suggest_float("rsi_exit_min", 60, 85)
    params["trigger_logic"]    = trial.suggest_categorical("trigger_logic", ["and", "or"])
    params["sl_mode"]              = trial.suggest_categorical("sl_mode", ["none","fixed","atr"])
    params["sl_pct"]               = trial.suggest_float("sl_pct", 0.01, 0.08)
    params["sl_atr_multiplier"]    = trial.suggest_float("sl_atr_multiplier", 1.0, 4.0)
    params["tp_mode"]              = trial.suggest_categorical("tp_mode", ["none","fixed","atr","rr"])
    params["tp_pct"]               = trial.suggest_float("tp_pct", 0.02, 0.15)
    params["tp_atr_multiplier"]    = trial.suggest_float("tp_atr_multiplier", 2.0, 8.0)
    params["tp_rr_ratio"]          = trial.suggest_float("tp_rr_ratio", 1.0, 4.0)
    params["trail_mode"]           = trial.suggest_categorical("trail_mode", ["none","fixed","atr"])
    params["trail_pct"]            = trial.suggest_float("trail_pct", 0.005, 0.05)
    params["trail_atr_multiplier"] = trial.suggest_float("trail_atr_multiplier", 0.5, 3.0)

    for col in selected:
        spec = FULL_INDICATOR_CATALOG.get(col)
        if spec is None:
            continue
        ic_val = (ic_scores or {}).get(col, 0.0)
        p_active = 0.3 + min(0.5, ic_val * 5.0)
        raw = trial.suggest_float(f"p_use_{col}", 0.0, 1.0)
        params[f"use_{col}"] = int(raw < p_active)
        if spec["type"] in ("osc", "gt", "lt", "band_pct") and "range" in spec:
            lo, hi = spec["range"]
            params[f"thresh_{col}"] = trial.suggest_float(f"thresh_{col}", lo, hi)

    if has_htf:
        params["use_htf_trend"]      = trial.suggest_categorical("use_htf_trend", [0, 1])
        params["use_htf_supertrend"] = trial.suggest_categorical("use_htf_supertrend", [0, 1])
        params["use_htf_rsi_filter"] = trial.suggest_categorical("use_htf_rsi_filter", [0, 1])
        params["htf_rsi_max"]        = trial.suggest_float("htf_rsi_max", 30, 70)
        params["use_htf_adx_filter"] = trial.suggest_categorical("use_htf_adx_filter", [0, 1])
        params["htf_adx_min"]        = trial.suggest_float("htf_adx_min", 15, 40)
        params["use_htf_ema200"]     = trial.suggest_categorical("use_htf_ema200", [0, 1])

    n_ltf_active = sum(1 for col in selected if params.get(f"use_{col}", 0))
    if n_ltf_active == 0:
        return (-999.0, -999.0)

    strategy = CatalogStrategy(params, selected)
    all_wfo  = []
    for df in dfs:
        if len(df) < 100:
            continue
        try:
            all_wfo.append(_run_wfo_for_trial(df, strategy, config, params))
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

    # Template selection: override indicators if template is set
    template_key = config.get("template", "free")
    template = STRATEGY_TEMPLATES.get(template_key, STRATEGY_TEMPLATES["free"])
    if template.get("indicators"):
        selected = [k for k in template["indicators"] if k in FULL_INDICATOR_CATALOG]
    else:
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

    primary_tf = timeframes[0] if timeframes else "1h"
    higher_tfs = timeframes[1:] if len(timeframes) > 1 else []

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
        return df

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
    regimes = [_detect_regime(df) for df in dfs[:10]]
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
        # Reconstruct use_X flags — they are NOT stored in trial.params because
        # they are derived (not suggested) inside the objective function.
        params   = _reconstruct_params(trial.params, selected, ic_scores)
        strategy = CatalogStrategy(params, selected)

        all_wfo  = []
        oos_trades_this = []   # all OOS trades from every pair × fold

        for df in dfs:
            try:
                wfo_r = _run_wfo_for_trial(df, strategy, config, params)
                all_wfo.append(wfo_r)
                # Collect OOS trade pnls for permutation test
                # _run_wfo_for_trial calls _walk_forward_backtest which uses fast_mode
                # We need the actual trade list. Re-run in non-fast mode for top 3 only.
                if rank <= 3:
                    try:
                        bt_kw = dict(
                            initial_capital=config.get("initial_capital", 10_000),
                            fee_rate=config.get("fee_rate", 0.001),
                            slippage=config.get("slippage", 0.0005),
                            position_size=config.get("position_size", 0.1),
                            fast_mode=False,
                        )
                        enriched = strategy.run(df.copy())
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

def _describe_rules(params: dict, selected: list) -> str:
    """Convert params dict to human-readable rule description."""
    trigger_logic = params.get("trigger_logic", "and")
    trigger_label = "ANY trigger (OR)" if trigger_logic == "or" else "ALL conditions (AND)"
    lines = [f"ENTRY CONDITIONS  [{trigger_label}]:"]
    active_mas   = []
    active_bands = []

    for col in selected:
        if not params.get(f"use_{col}", 0):
            continue
        spec = FULL_INDICATOR_CATALOG.get(col)
        if not spec:
            continue
        kind = spec["type"]

        t_tag = " [trigger]" if trigger_logic == "or" and kind in ("osc", "band_pct", "band_lower", "cdl", "cross_above", "cross_below") else ""
        if kind in ("osc", "band_pct", "lt"):
            lines.append(f"  • {col} < {params[f'thresh_{col}']:.4g}{t_tag}")
        elif kind == "gt":
            lines.append(f"  • {col} > {params[f'thresh_{col}']:.4g}")
        elif kind == "sign":
            lines.append(f"  • {col} > 0")
        elif kind == "flag":
            lines.append(f"  • {col} = Bullish")
        elif kind == "cdl":
            lines.append(f"  • {spec.get('label', col)} pattern{t_tag}")
        elif kind == "ma":
            lines.append(f"  • close > {col}")
            active_mas.append((_ma_period(col), col))
        elif kind == "band_lower":
            lines.append(f"  • close < {col}  (oversold below lower band){t_tag}")
            active_bands.append(col)
        elif kind in ("cross_above", "cross_below"):
            lines.append(f"  • {spec['label']}  [crossover — 1 bar]{t_tag}")

    # Derived cross-conditions
    active_mas.sort()
    if len(active_mas) >= 2:
        lines.append(f"  • {active_mas[0][1]} > {active_mas[-1][1]}  [MA alignment — derived]")
    if active_bands and active_mas:
        anchor = active_mas[-1][1]
        for band_col in active_bands:
            lines.append(f"  • {band_col} > {anchor}  [floor above trend — derived]")

    # HTF filters
    if params.get("use_htf_trend"):
        lines.append("  • [HTF] Price > HTF EMA50 (trend aligned)")
    if params.get("use_htf_supertrend"):
        lines.append("  • [HTF] Supertrend Bullish on higher TF")
    if params.get("use_htf_rsi_filter"):
        lines.append(f"  • [HTF] RSI < {params.get('htf_rsi_max', 60):.1f} on higher TF")
    if params.get("use_htf_adx_filter"):
        lines.append(f"  • [HTF] ADX > {params.get('htf_adx_min', 20):.1f} on higher TF")
    if params.get("use_htf_ema200"):
        lines.append("  • [HTF] Price > EMA(200) on higher TF")

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
