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
        files = sorted(session_dir.glob(f"*_{primary_tf}.parquet"))
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

    return catalog if catalog else FULL_INDICATOR_CATALOG


# EMA period embedded in column names — used to sort active MAs for cross-conditions
def _ma_period(col: str) -> int:
    """Extract numeric period from EMA_XX column name."""
    try:
        return int(col.split("_")[1])
    except (IndexError, ValueError):
        return 0


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
        cond = pd.Series(True, index=df.index)

        active_mas   = []   # (period_int, col) for active "ma" entries
        active_bands = []   # col names for active "band_lower" entries

        for col in self._selected:
            if not p.get(f"use_{col}", 0):
                continue
            spec = FULL_INDICATOR_CATALOG.get(col)
            if spec is None or col not in df.columns:
                continue

            kind = spec["type"]

            if kind == "osc" or kind == "band_pct" or kind == "lt":
                cond &= df[col] < p[f"thresh_{col}"]

            elif kind == "gt":
                cond &= df[col] > p[f"thresh_{col}"]

            elif kind == "sign":
                cond &= df[col] > 0

            elif kind == "flag":
                cond &= df[col] == 1

            elif kind == "ma":
                cond &= df["close"] > df[col]
                active_mas.append((_ma_period(col), col))

            elif kind == "band_lower":
                cond &= df["close"] < df[col]
                active_bands.append(col)

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
    """
    Optuna objective.

    For each selected indicator:
      • suggest use_<col>    (0/1 — active or not this trial)
      • suggest thresh_<col> (float — only for osc/gt/lt/band_pct types)

    Cross-conditions between price-scale indicators (MA alignment, band vs MA)
    are derived automatically inside CatalogStrategy — no extra parameters needed.
    """
    params = {}

    # Exit parameters (always tuned regardless of indicator selection)
    params["rsi_exit_period"] = trial.suggest_categorical("rsi_exit_period", [7, 14, 21])
    params["rsi_exit_min"]    = trial.suggest_float("rsi_exit_min", 60, 85)

    # Per-indicator parameters
    for col in selected:
        spec = FULL_INDICATOR_CATALOG.get(col)
        if spec is None:
            continue
        params[f"use_{col}"] = trial.suggest_categorical(f"use_{col}", [0, 1])
        if spec["type"] in ("osc", "gt", "lt", "band_pct") and "range" in spec:
            lo, hi = spec["range"]
            params[f"thresh_{col}"] = trial.suggest_float(f"thresh_{col}", lo, hi)

    # HTF filters (only when HTF data is present)
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
    # Keep only keys that exist in the full catalog (static + auto-classified)
    selected = [k for k in selected if k in FULL_INDICATOR_CATALOG]
    if not selected:
        selected = DEFAULT_INDICATORS

    def progress(p, total, msg):
        m.update_task(db_path, task_id, progress=p, total=total, message=msg)

    progress(0, n_trials, "Loading data for Algo Finder…")

    pairs  = m.list_pairs(db_path, session_id)
    active = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]

    # Filter to user-selected pairs when specified
    selected_pairs = config.get("selected_pairs") or []
    if selected_pairs:
        active = [p for p in active if p["symbol"] in selected_pairs]
    if not active:
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
        # Skip recomputation when Phase 5 already stored indicators in the parquet
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

    sample_pairs = active[:50]   # cap at 50; user can narrow via per-pair selection
    workers = min(os.cpu_count() or 4, len(sample_pairs), 8)
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
        FULL_INDICATOR_CATALOG[k]["label"] for k in selected if k in FULL_INDICATOR_CATALOG
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
    """Convert params dict to human-readable rule description."""
    lines = ["ENTRY CONDITIONS:"]
    active_mas   = []
    active_bands = []

    for col in selected:
        if not params.get(f"use_{col}", 0):
            continue
        spec = FULL_INDICATOR_CATALOG.get(col)
        if not spec:
            continue
        kind = spec["type"]

        if kind in ("osc", "band_pct", "lt"):
            lines.append(f"  • {col} < {params[f'thresh_{col}']:.4g}")
        elif kind == "gt":
            lines.append(f"  • {col} > {params[f'thresh_{col}']:.4g}")
        elif kind == "sign":
            lines.append(f"  • {col} > 0")
        elif kind == "flag":
            lines.append(f"  • {col} = Bullish")
        elif kind == "ma":
            lines.append(f"  • close > {col}")
            active_mas.append((_ma_period(col), col))
        elif kind == "band_lower":
            lines.append(f"  • close < {col}  (oversold below lower band)")
            active_bands.append(col)

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
