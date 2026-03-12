from flask import Blueprint, render_template

bp = Blueprint("indicators_lib", __name__)

# ── Master indicator catalog ───────────────────────────────────────────────────
# "col" = the exact DataFrame column name written by populate_indicators().
# "name" = display label. "desc" = tooltip / description.
INDICATOR_CATALOG = [

    # ══════════════════════════════════════════════════════════════════════════
    # TREND — moving averages, trend direction, crossovers
    # ══════════════════════════════════════════════════════════════════════════

    # ── Standard EMAs ──────────────────────────────────────────────────────────
    {"name": "EMA_8",           "category": "Trend",   "col": "EMA_8",           "desc": "Exponential MA (8)"},
    {"name": "EMA_13",          "category": "Trend",   "col": "EMA_13",          "desc": "Exponential MA (13)"},
    {"name": "EMA_20",          "category": "Trend",   "col": "EMA_20",          "desc": "Exponential MA (20)"},
    {"name": "EMA_21",          "category": "Trend",   "col": "EMA_21",          "desc": "Exponential MA (21)"},
    {"name": "EMA_50",          "category": "Trend",   "col": "EMA_50",          "desc": "Exponential MA (50)"},
    {"name": "EMA_100",         "category": "Trend",   "col": "EMA_100",         "desc": "Exponential MA (100)"},
    {"name": "EMA_200",         "category": "Trend",   "col": "EMA_200",         "desc": "Exponential MA (200)"},
    # ── SMAs ───────────────────────────────────────────────────────────────────
    {"name": "SMA_8",           "category": "Trend",   "col": "SMA_8",           "desc": "Simple MA (8)"},
    {"name": "SMA_13",          "category": "Trend",   "col": "SMA_13",          "desc": "Simple MA (13)"},
    {"name": "SMA_20",          "category": "Trend",   "col": "SMA_20",          "desc": "Simple MA (20)"},
    {"name": "SMA_21",          "category": "Trend",   "col": "SMA_21",          "desc": "Simple MA (21)"},
    {"name": "SMA_50",          "category": "Trend",   "col": "SMA_50",          "desc": "Simple MA (50)"},
    {"name": "SMA_100",         "category": "Trend",   "col": "SMA_100",         "desc": "Simple MA (100)"},
    {"name": "SMA_200",         "category": "Trend",   "col": "SMA_200",         "desc": "Simple MA (200)"},
    # ── WMAs ───────────────────────────────────────────────────────────────────
    {"name": "WMA_8",           "category": "Trend",   "col": "WMA_8",           "desc": "Weighted MA (8)"},
    {"name": "WMA_20",          "category": "Trend",   "col": "WMA_20",          "desc": "Weighted MA (20)"},
    {"name": "WMA_50",          "category": "Trend",   "col": "WMA_50",          "desc": "Weighted MA (50)"},
    # ── Hull MA ────────────────────────────────────────────────────────────────
    {"name": "HMA_9",           "category": "Trend",   "col": "HMA_9",           "desc": "Hull MA (9) — low lag"},
    {"name": "HMA_20",          "category": "Trend",   "col": "HMA_20",          "desc": "Hull MA (20)"},
    {"name": "HMA_50",          "category": "Trend",   "col": "HMA_50",          "desc": "Hull MA (50)"},
    # ── DEMA / TEMA ────────────────────────────────────────────────────────────
    {"name": "DEMA_9",          "category": "Trend",   "col": "DEMA_9",          "desc": "Double EMA (9) — less lag than EMA"},
    {"name": "DEMA_21",         "category": "Trend",   "col": "DEMA_21",         "desc": "Double EMA (21)"},
    {"name": "DEMA_50",         "category": "Trend",   "col": "DEMA_50",         "desc": "Double EMA (50)"},
    {"name": "TEMA_9",          "category": "Trend",   "col": "TEMA_9",          "desc": "Triple EMA (9)"},
    {"name": "TEMA_21",         "category": "Trend",   "col": "TEMA_21",         "desc": "Triple EMA (21)"},
    {"name": "TEMA_50",         "category": "Trend",   "col": "TEMA_50",         "desc": "Triple EMA (50)"},
    # ── T3 ─────────────────────────────────────────────────────────────────────
    {"name": "T3_5",            "category": "Trend",   "col": "T3_5",            "desc": "T3 Triple EMA (5) — very smooth, low lag"},
    {"name": "T3_10",           "category": "Trend",   "col": "T3_10",           "desc": "T3 Triple EMA (10)"},
    # ── Triangular MA ──────────────────────────────────────────────────────────
    {"name": "TRIMA_20",        "category": "Trend",   "col": "TRIMA_20",        "desc": "Triangular MA (20) — double-smoothed SMA"},
    {"name": "TRIMA_50",        "category": "Trend",   "col": "TRIMA_50",        "desc": "Triangular MA (50)"},
    # ── Zero-Lag EMA ───────────────────────────────────────────────────────────
    {"name": "ZLEMA_20",        "category": "Trend",   "col": "ZLEMA_20",        "desc": "Zero-Lag EMA (20) — corrects for lag"},
    {"name": "ZLEMA_50",        "category": "Trend",   "col": "ZLEMA_50",        "desc": "Zero-Lag EMA (50)"},
    # ── Arnaud Legoux MA ───────────────────────────────────────────────────────
    {"name": "ALMA_9",          "category": "Trend",   "col": "ALMA_9",          "desc": "Arnaud Legoux MA (9, sigma=6, offset=0.85)"},
    {"name": "ALMA_21",         "category": "Trend",   "col": "ALMA_21",         "desc": "Arnaud Legoux MA (21)"},
    # ── Adaptive / Special MAs ─────────────────────────────────────────────────
    {"name": "KAMA",            "category": "Trend",   "col": "KAMA",            "desc": "Kaufman Adaptive MA — adapts speed to market noise"},
    {"name": "MCGD_14",         "category": "Trend",   "col": "MCGD_14",         "desc": "McGinley Dynamic (14) — auto-adjusts smoothing"},
    {"name": "VIDYA_14",        "category": "Trend",   "col": "VIDYA_14",        "desc": "Variable Index Dynamic Average (14)"},
    {"name": "HWMA",            "category": "Trend",   "col": "HWMA",            "desc": "Holt-Winters MA — triple exponential smoothing"},
    {"name": "SWMA",            "category": "Trend",   "col": "SWMA",            "desc": "Symmetric Weighted MA (4-bar triangular weights)"},
    {"name": "FWMA_10",         "category": "Trend",   "col": "FWMA_10",         "desc": "Fibonacci Weighted MA (10)"},
    {"name": "PWMA_10",         "category": "Trend",   "col": "PWMA_10",         "desc": "Pascal Weighted MA (10)"},
    {"name": "VWMA_20",         "category": "Trend",   "col": "VWMA_20",         "desc": "Volume-Weighted MA (20)"},
    {"name": "VWMA_50",         "category": "Trend",   "col": "VWMA_50",         "desc": "Volume-Weighted MA (50)"},
    # ── MA distances & crosses ─────────────────────────────────────────────────
    {"name": "close_vs_EMA_20", "category": "Trend",   "col": "close_vs_EMA_20", "desc": "Distance of close from EMA20 in %"},
    {"name": "close_vs_EMA_50", "category": "Trend",   "col": "close_vs_EMA_50", "desc": "Distance of close from EMA50 in %"},
    {"name": "close_vs_EMA_200","category": "Trend",   "col": "close_vs_EMA_200","desc": "Distance of close from EMA200 in %"},
    {"name": "ema_20_50_cross", "category": "Trend",   "col": "ema_20_50_cross", "desc": "EMA20/50 cross: +1=golden cross, -1=death cross"},
    {"name": "ema_50_200_cross","category": "Trend",   "col": "ema_50_200_cross","desc": "EMA50/200 cross: +1=golden cross, -1=death cross"},
    {"name": "ema_aligned_bull","category": "Trend",   "col": "ema_aligned_bull","desc": "1 if EMA20 > EMA50 > EMA200 (full bull alignment)"},
    {"name": "EMA50_slope",     "category": "Trend",   "col": "EMA50_slope",     "desc": "EMA50 5-bar slope as % — quantifies trend strength"},
    {"name": "AMAT_fast",       "category": "Trend",   "col": "AMAT_fast",       "desc": "Absolute MA Trend: 1 if EMA8 > EMA21, else 0"},
    # ── MACD ───────────────────────────────────────────────────────────────────
    {"name": "MACD",            "category": "Trend",   "col": "MACD",            "desc": "MACD line (EMA12 − EMA26)"},
    {"name": "MACD_signal",     "category": "Trend",   "col": "MACD_signal",     "desc": "MACD signal line (9-EMA of MACD)"},
    {"name": "MACD_hist",       "category": "Trend",   "col": "MACD_hist",       "desc": "MACD histogram (MACD − signal)"},
    {"name": "MACD_cross",      "category": "Trend",   "col": "MACD_cross",      "desc": "MACD/signal cross: +1=bullish, -1=bearish, 0=none"},
    # ── ADX / DMI ──────────────────────────────────────────────────────────────
    {"name": "ADX_7",           "category": "Trend",   "col": "ADX_7",           "desc": "ADX (7) — fast trend strength, 0–100"},
    {"name": "ADX_14",          "category": "Trend",   "col": "ADX_14",          "desc": "ADX (14) — standard trend strength, >25 = trending"},
    {"name": "ADX_21",          "category": "Trend",   "col": "ADX_21",          "desc": "ADX (21) — slow trend strength"},
    {"name": "DMP_14",          "category": "Trend",   "col": "DMP_14",          "desc": "DMI+ (14) — positive directional indicator"},
    {"name": "DMN_14",          "category": "Trend",   "col": "DMN_14",          "desc": "DMI− (14) — negative directional indicator"},
    {"name": "DMP_7",           "category": "Trend",   "col": "DMP_7",           "desc": "DMI+ (7)"},
    {"name": "DMN_7",           "category": "Trend",   "col": "DMN_7",           "desc": "DMI− (7)"},
    {"name": "DMP_21",          "category": "Trend",   "col": "DMP_21",          "desc": "DMI+ (21)"},
    {"name": "DMN_21",          "category": "Trend",   "col": "DMN_21",          "desc": "DMI− (21)"},
    # ── Aroon ──────────────────────────────────────────────────────────────────
    {"name": "AROON_up",        "category": "Trend",   "col": "AROON_up",        "desc": "Aroon Up (25) — 0–100, how recent was the high"},
    {"name": "AROON_down",      "category": "Trend",   "col": "AROON_down",      "desc": "Aroon Down (25) — 0–100, how recent was the low"},
    {"name": "AROON_osc",       "category": "Trend",   "col": "AROON_osc",       "desc": "Aroon Oscillator (Up − Down), +100 to −100"},
    {"name": "AROONOSC_25",     "category": "Trend",   "col": "AROONOSC_25",     "desc": "Aroon Oscillator (25 period)"},
    # ── Vortex ─────────────────────────────────────────────────────────────────
    {"name": "VI_pos",          "category": "Trend",   "col": "VI_pos",          "desc": "Vortex Indicator positive (14)"},
    {"name": "VI_neg",          "category": "Trend",   "col": "VI_neg",          "desc": "Vortex Indicator negative (14)"},
    # ── Ichimoku ───────────────────────────────────────────────────────────────
    {"name": "ICH_tenkan",      "category": "Trend",   "col": "ICH_tenkan",      "desc": "Ichimoku Tenkan-sen (conversion line, 9)"},
    {"name": "ICH_kijun",       "category": "Trend",   "col": "ICH_kijun",       "desc": "Ichimoku Kijun-sen (base line, 26)"},
    {"name": "ICH_senkou_a",    "category": "Trend",   "col": "ICH_senkou_a",    "desc": "Ichimoku Senkou Span A (cloud top)"},
    {"name": "ICH_senkou_b",    "category": "Trend",   "col": "ICH_senkou_b",    "desc": "Ichimoku Senkou Span B (cloud bottom)"},
    {"name": "ICH_above_cloud", "category": "Trend",   "col": "ICH_above_cloud", "desc": "1 if close is above both Senkou spans (bullish)"},
    {"name": "ICH_below_cloud", "category": "Trend",   "col": "ICH_below_cloud", "desc": "1 if close is below both Senkou spans (bearish)"},
    # ── Parabolic SAR ──────────────────────────────────────────────────────────
    {"name": "PSAR_up",         "category": "Trend",   "col": "PSAR_up",         "desc": "Parabolic SAR value when in uptrend"},
    {"name": "PSAR_down",       "category": "Trend",   "col": "PSAR_down",       "desc": "Parabolic SAR value when in downtrend"},
    {"name": "PSAR_dir",        "category": "Trend",   "col": "PSAR_dir",        "desc": "PSAR direction: +1=bullish, −1=bearish"},
    # ── Supertrend ─────────────────────────────────────────────────────────────
    {"name": "SUPERT_dir",      "category": "Trend",   "col": "SUPERT_dir",      "desc": "Supertrend dir (10,3) — +1=bullish, −1=bearish"},
    {"name": "SUPERT_10_3",     "category": "Trend",   "col": "SUPERT_10_3",     "desc": "Supertrend line value (period=10, mult=3.0)"},
    {"name": "SUPERT_dir_10_3", "category": "Trend",   "col": "SUPERT_dir_10_3", "desc": "Supertrend direction (10, 3.0)"},
    {"name": "SUPERT_dir_14_2", "category": "Trend",   "col": "SUPERT_dir_14_2", "desc": "Supertrend direction (14, 2.0) — tighter"},
    {"name": "SUPERT_dir_7_3",  "category": "Trend",   "col": "SUPERT_dir_7_3",  "desc": "Supertrend direction (7, 3.0) — fast"},
    {"name": "SUPERT_dir_10_2", "category": "Trend",   "col": "SUPERT_dir_10_2", "desc": "Supertrend direction (10, 2.0)"},
    {"name": "SUPERT_dir_14_3", "category": "Trend",   "col": "SUPERT_dir_14_3", "desc": "Supertrend direction (14, 3.0)"},
    {"name": "SUPERT_dir_20_2", "category": "Trend",   "col": "SUPERT_dir_20_2", "desc": "Supertrend direction (20, 2.0) — slow"},
    {"name": "PMAX_dir",        "category": "Trend",   "col": "PMAX_dir",        "desc": "PMAX direction — Supertrend using VWMA base"},
    # ── Williams Alligator ─────────────────────────────────────────────────────
    {"name": "ALLIGATOR_JAW",   "category": "Trend",   "col": "ALLIGATOR_JAW",   "desc": "Williams Alligator Jaw — SMMA(13), slowest line"},
    {"name": "ALLIGATOR_TEETH", "category": "Trend",   "col": "ALLIGATOR_TEETH", "desc": "Williams Alligator Teeth — SMMA(8)"},
    {"name": "ALLIGATOR_LIPS",  "category": "Trend",   "col": "ALLIGATOR_LIPS",  "desc": "Williams Alligator Lips — SMMA(5), fastest line"},
    {"name": "ALLIGATOR_BULL",  "category": "Trend",   "col": "ALLIGATOR_BULL",  "desc": "1 if Lips > Teeth > Jaw (alligator opening bullish)"},
    {"name": "ALLIGATOR_BEAR",  "category": "Trend",   "col": "ALLIGATOR_BEAR",  "desc": "1 if Lips < Teeth < Jaw (alligator opening bearish)"},
    # ── Guppy GMMA ─────────────────────────────────────────────────────────────
    {"name": "GMMA_S3",         "category": "Trend",   "col": "GMMA_S3",         "desc": "Guppy GMMA short EMA (3)"},
    {"name": "GMMA_S5",         "category": "Trend",   "col": "GMMA_S5",         "desc": "Guppy GMMA short EMA (5)"},
    {"name": "GMMA_S8",         "category": "Trend",   "col": "GMMA_S8",         "desc": "Guppy GMMA short EMA (8)"},
    {"name": "GMMA_S10",        "category": "Trend",   "col": "GMMA_S10",        "desc": "Guppy GMMA short EMA (10)"},
    {"name": "GMMA_S12",        "category": "Trend",   "col": "GMMA_S12",        "desc": "Guppy GMMA short EMA (12)"},
    {"name": "GMMA_S15",        "category": "Trend",   "col": "GMMA_S15",        "desc": "Guppy GMMA short EMA (15)"},
    {"name": "GMMA_L30",        "category": "Trend",   "col": "GMMA_L30",        "desc": "Guppy GMMA long EMA (30)"},
    {"name": "GMMA_L35",        "category": "Trend",   "col": "GMMA_L35",        "desc": "Guppy GMMA long EMA (35)"},
    {"name": "GMMA_L40",        "category": "Trend",   "col": "GMMA_L40",        "desc": "Guppy GMMA long EMA (40)"},
    {"name": "GMMA_L45",        "category": "Trend",   "col": "GMMA_L45",        "desc": "Guppy GMMA long EMA (45)"},
    {"name": "GMMA_L50",        "category": "Trend",   "col": "GMMA_L50",        "desc": "Guppy GMMA long EMA (50)"},
    {"name": "GMMA_L60",        "category": "Trend",   "col": "GMMA_L60",        "desc": "Guppy GMMA long EMA (60)"},
    {"name": "GMMA_BULL",       "category": "Trend",   "col": "GMMA_BULL",       "desc": "1 if GMMA short group > long group (bull phase)"},
    # ── Hilo Activator ─────────────────────────────────────────────────────────
    {"name": "HILO_HIGH",       "category": "Trend",   "col": "HILO_HIGH",       "desc": "Hilo Activator high band (SMA 13 of highs)"},
    {"name": "HILO_LOW",        "category": "Trend",   "col": "HILO_LOW",        "desc": "Hilo Activator low band (SMA 21 of lows)"},
    {"name": "HILO_dir",        "category": "Trend",   "col": "HILO_dir",        "desc": "Hilo direction: +1=above high band, −1=below low"},
    # ── Other trend oscillators ────────────────────────────────────────────────
    {"name": "DPO_20",          "category": "Trend",   "col": "DPO_20",          "desc": "Detrended Price Oscillator (20) — isolates cycles"},
    {"name": "KST",             "category": "Trend",   "col": "KST",             "desc": "Know Sure Thing oscillator"},
    {"name": "KST_signal",      "category": "Trend",   "col": "KST_signal",      "desc": "KST signal line"},
    {"name": "TRIX_15",         "category": "Trend",   "col": "TRIX_15",         "desc": "TRIX: 1-day ROC of triple-smoothed EMA (15)"},
    {"name": "MASS_INDEX",      "category": "Trend",   "col": "MASS_INDEX",      "desc": "Mass Index (9/25) — volatility reversal signal"},
    {"name": "TREND_STR_14",    "category": "Trend",   "col": "TREND_STR_14",    "desc": "Trend strength: % of last 14 bars that closed up"},
    {"name": "TSF_14",          "category": "Trend",   "col": "TSF_14",          "desc": "Time Series Forecast (14) — linear regression projection"},

    # ══════════════════════════════════════════════════════════════════════════
    # MOMENTUM — oscillators, rate of change, strength indicators
    # ══════════════════════════════════════════════════════════════════════════

    # ── RSI variants ───────────────────────────────────────────────────────────
    {"name": "RSI_7",           "category": "Momentum","col": "RSI_7",           "desc": "RSI (7) — fast, reactive"},
    {"name": "RSI_14",          "category": "Momentum","col": "RSI_14",          "desc": "RSI (14) — standard, >70 overbought, <30 oversold"},
    {"name": "RSI_21",          "category": "Momentum","col": "RSI_21",          "desc": "RSI (21) — slow, fewer false signals"},
    {"name": "RSX_14",          "category": "Momentum","col": "RSX_14",          "desc": "RSX (14) — smoother RSI with less noise"},
    {"name": "RSI_DIVERG",      "category": "Momentum","col": "RSI_DIVERG",      "desc": "1 when price makes new high but RSI doesn't (bearish divergence)"},
    {"name": "CRSI",            "category": "Momentum","col": "CRSI",            "desc": "Connors RSI — composite of RSI(3), streak RSI, percentile rank"},
    # ── Stochastic ─────────────────────────────────────────────────────────────
    {"name": "STOCH_K",         "category": "Momentum","col": "STOCH_K",         "desc": "Stochastic %K (14,3)"},
    {"name": "STOCH_D",         "category": "Momentum","col": "STOCH_D",         "desc": "Stochastic %D — signal line (3-period SMA of %K)"},
    {"name": "STOCHRSI_K",      "category": "Momentum","col": "STOCHRSI_K",      "desc": "StochRSI %K (14, smooth1=3, smooth2=3)"},
    {"name": "STOCHRSI_D",      "category": "Momentum","col": "STOCHRSI_D",      "desc": "StochRSI %D signal"},
    {"name": "STOCHRSI_K_smooth","category":"Momentum","col": "STOCHRSI_K_smooth","desc": "StochRSI %K with extra 3-bar smoothing"},
    {"name": "KDJ_K",           "category": "Momentum","col": "KDJ_K",           "desc": "KDJ K line (stochastic-based, signal=3)"},
    {"name": "KDJ_D",           "category": "Momentum","col": "KDJ_D",           "desc": "KDJ D line"},
    {"name": "KDJ_J",           "category": "Momentum","col": "KDJ_J",           "desc": "KDJ J line (3K − 2D) — amplifies reversals"},
    # ── ROC / Momentum ─────────────────────────────────────────────────────────
    {"name": "ROC_5",           "category": "Momentum","col": "ROC_5",           "desc": "Rate of Change (5) in %"},
    {"name": "ROC_10",          "category": "Momentum","col": "ROC_10",          "desc": "Rate of Change (10) in %"},
    {"name": "ROC_14",          "category": "Momentum","col": "ROC_14",          "desc": "Rate of Change (14) in %"},
    {"name": "ROC_20",          "category": "Momentum","col": "ROC_20",          "desc": "Rate of Change (20) in %"},
    {"name": "ROC_30",          "category": "Momentum","col": "ROC_30",          "desc": "Rate of Change (30) in %"},
    {"name": "ROCP_10",         "category": "Momentum","col": "ROCP_10",         "desc": "ROC percentage — (close−prev)/prev"},
    {"name": "ROCR_10",         "category": "Momentum","col": "ROCR_10",         "desc": "ROC ratio — close/prev"},
    {"name": "ROCR100_10",      "category": "Momentum","col": "ROCR100_10",      "desc": "ROC ratio ×100 scale"},
    {"name": "MOM_10",          "category": "Momentum","col": "MOM_10",          "desc": "Momentum (10) — close − close[10]"},
    {"name": "MOM_20",          "category": "Momentum","col": "MOM_20",          "desc": "Momentum (20)"},
    {"name": "close_pct_change","category": "Momentum","col": "close_pct_change","desc": "Bar-over-bar close percent change"},
    # ── Classic oscillators ────────────────────────────────────────────────────
    {"name": "CCI_20",          "category": "Momentum","col": "CCI_20",          "desc": "Commodity Channel Index (20), ±100 = extreme"},
    {"name": "CMO_14",          "category": "Momentum","col": "CMO_14",          "desc": "Chande Momentum Oscillator (14), range −100/+100"},
    {"name": "FISHER",          "category": "Momentum","col": "FISHER",          "desc": "Fisher Transform (9) — ±2.5 signal reversals"},
    {"name": "AO",              "category": "Momentum","col": "AO",              "desc": "Awesome Oscillator (SMA5 − SMA34 of midprice)"},
    {"name": "UO",              "category": "Momentum","col": "UO",              "desc": "Ultimate Oscillator (7,14,28 weighted) — 0 to 100"},
    {"name": "WILLR_14",        "category": "Momentum","col": "WILLR_14",        "desc": "Williams %R (14), range −100 to 0, <−80 oversold"},
    {"name": "APO_12_26",       "category": "Momentum","col": "APO_12_26",       "desc": "Absolute Price Oscillator (EMA12 − EMA26)"},
    {"name": "PPO",             "category": "Momentum","col": "PPO",             "desc": "Percentage Price Oscillator"},
    {"name": "PPO_signal",      "category": "Momentum","col": "PPO_signal",      "desc": "PPO signal line"},
    {"name": "PPO_hist",        "category": "Momentum","col": "PPO_hist",        "desc": "PPO histogram"},
    # ── Advanced momentum oscillators ──────────────────────────────────────────
    {"name": "TSI_13_25",       "category": "Momentum","col": "TSI_13_25",       "desc": "True Strength Index (fast=13, slow=25) — double-smoothed momentum"},
    {"name": "SMI",             "category": "Momentum","col": "SMI",             "desc": "SMI Ergodic Indicator (5,20) — centered oscillator"},
    {"name": "SMI_signal",      "category": "Momentum","col": "SMI_signal",      "desc": "SMI signal line"},
    {"name": "STC",             "category": "Momentum","col": "STC",             "desc": "Schaff Trend Cycle — fast cycle oscillator, 0–100"},
    {"name": "QQE_LINE",        "category": "Momentum","col": "QQE_LINE",        "desc": "QQE smoothed RSI line"},
    {"name": "QQE_HIST",        "category": "Momentum","col": "QQE_HIST",        "desc": "QQE fast ATR band — use for crossover signals"},
    {"name": "RVGI_14",         "category": "Momentum","col": "RVGI_14",         "desc": "Relative Vigor Index (14) — close−open vs range"},
    {"name": "CFO_9",           "category": "Momentum","col": "CFO_9",           "desc": "Chande Forecast Oscillator (9) — price vs LinReg(9)"},
    {"name": "CG_10",           "category": "Momentum","col": "CG_10",           "desc": "Center of Gravity (10) — leading reversal signal"},
    {"name": "INERTIA_20",      "category": "Momentum","col": "INERTIA_20",      "desc": "Inertia (20) — linear regression of RVI smoothed"},
    {"name": "BIAS_6",          "category": "Momentum","col": "BIAS_6",          "desc": "Bias (6) — distance from EMA6 in %"},
    {"name": "BIAS_14",         "category": "Momentum","col": "BIAS_14",         "desc": "Bias (14)"},
    {"name": "BIAS_26",         "category": "Momentum","col": "BIAS_26",         "desc": "Bias (26)"},
    {"name": "PSL_12",          "category": "Momentum","col": "PSL_12",          "desc": "Psychological Line (12) — % of bars that closed up"},
    {"name": "COPPOCK",         "category": "Momentum","col": "COPPOCK",         "desc": "Coppock Curve — long-term momentum buy signal"},
    {"name": "TTM_TREND_6",     "category": "Momentum","col": "TTM_TREND_6",     "desc": "TTM Trend (6) — +1 bull, −1 bear bars"},
    {"name": "PMAX_dir",        "category": "Momentum","col": "PMAX_dir",        "desc": "PMAX direction using VWMA-based Supertrend"},

    # ══════════════════════════════════════════════════════════════════════════
    # VOLATILITY — bands, ATR, range, squeeze, dispersion
    # ══════════════════════════════════════════════════════════════════════════

    # ── ATR / NATR ─────────────────────────────────────────────────────────────
    {"name": "ATR_7",           "category": "Volatility","col": "ATR_7",         "desc": "Average True Range (7) — fast"},
    {"name": "ATR_14",          "category": "Volatility","col": "ATR_14",        "desc": "Average True Range (14) — standard volatility"},
    {"name": "ATR_21",          "category": "Volatility","col": "ATR_21",        "desc": "Average True Range (21) — slow"},
    {"name": "NATR_7",          "category": "Volatility","col": "NATR_7",        "desc": "Normalised ATR (7) as % of close"},
    {"name": "NATR_14",         "category": "Volatility","col": "NATR_14",       "desc": "Normalised ATR (14) — comparable across assets"},
    {"name": "NATR_21",         "category": "Volatility","col": "NATR_21",       "desc": "Normalised ATR (21)"},
    {"name": "TRANGE",          "category": "Volatility","col": "TRANGE",        "desc": "True Range — max(H−L, |H−Cprev|, |L−Cprev|)"},
    {"name": "VOL_RATIO_14",    "category": "Volatility","col": "VOL_RATIO_14",  "desc": "ATR14 / 100-bar average ATR — above 1 = elevated vol"},
    # ── Historical Volatility ──────────────────────────────────────────────────
    {"name": "HV_10",           "category": "Volatility","col": "HV_10",        "desc": "Historical Volatility (10-bar, annualised log-return std)"},
    {"name": "HV_20",           "category": "Volatility","col": "HV_20",        "desc": "Historical Volatility (20-bar)"},
    {"name": "HV_30",           "category": "Volatility","col": "HV_30",        "desc": "Historical Volatility (30-bar)"},
    {"name": "HV_60",           "category": "Volatility","col": "HV_60",        "desc": "Historical Volatility (60-bar)"},
    {"name": "HIGH_VOL_REGIME", "category": "Volatility","col": "HIGH_VOL_REGIME","desc": "1 if current ATR14 is in top 25% of last 100 bars"},
    # ── Bollinger Bands ────────────────────────────────────────────────────────
    {"name": "BB_upper_14",     "category": "Volatility","col": "BB_upper_14",  "desc": "Bollinger Band upper (14, 2σ)"},
    {"name": "BB_mid_14",       "category": "Volatility","col": "BB_mid_14",    "desc": "Bollinger Band middle (14 SMA)"},
    {"name": "BB_lower_14",     "category": "Volatility","col": "BB_lower_14",  "desc": "Bollinger Band lower (14, 2σ)"},
    {"name": "BB_pct_14",       "category": "Volatility","col": "BB_pct_14",    "desc": "Bollinger %B (14): 0=lower, 1=upper"},
    {"name": "BB_width_14",     "category": "Volatility","col": "BB_width_14",  "desc": "Bollinger Band width (14) — normalised bandwidth"},
    {"name": "BB_upper_20",     "category": "Volatility","col": "BB_upper_20",  "desc": "Bollinger Band upper (20, 2σ)"},
    {"name": "BB_mid_20",       "category": "Volatility","col": "BB_mid_20",    "desc": "Bollinger Band middle (20 SMA)"},
    {"name": "BB_lower_20",     "category": "Volatility","col": "BB_lower_20",  "desc": "Bollinger Band lower (20, 2σ)"},
    {"name": "BB_pct_20",       "category": "Volatility","col": "BB_pct_20",    "desc": "Bollinger %B (20): 0=lower, 1=upper"},
    {"name": "BB_width_20",     "category": "Volatility","col": "BB_width_20",  "desc": "Bollinger Band width (20) — proxy for volatility regime"},
    {"name": "BB_SQUEEZE",      "category": "Volatility","col": "BB_SQUEEZE",   "desc": "1 if BB bands are inside Keltner channels (energy build-up)"},
    {"name": "SQUEEZE_HIST",    "category": "Volatility","col": "SQUEEZE_HIST", "desc": "Squeeze momentum: close vs midpoint of KC (positive=bullish)"},
    # ── Keltner / Donchian ────────────────────────────────────────────────────
    {"name": "KC_upper",        "category": "Volatility","col": "KC_upper",     "desc": "Keltner Channel upper (20, ATR10)"},
    {"name": "KC_lower",        "category": "Volatility","col": "KC_lower",     "desc": "Keltner Channel lower"},
    {"name": "KC_middle",       "category": "Volatility","col": "KC_middle",    "desc": "Keltner Channel middle (EMA20)"},
    {"name": "KC_WIDTH",        "category": "Volatility","col": "KC_WIDTH",     "desc": "Keltner Channel width as % of close"},
    {"name": "DC_upper",        "category": "Volatility","col": "DC_upper",     "desc": "Donchian Channel upper (20-bar high)"},
    {"name": "DC_lower",        "category": "Volatility","col": "DC_lower",     "desc": "Donchian Channel lower (20-bar low)"},
    {"name": "DC_middle",       "category": "Volatility","col": "DC_middle",    "desc": "Donchian Channel middle"},
    {"name": "DC_WIDTH",        "category": "Volatility","col": "DC_WIDTH",     "desc": "Donchian Channel width as % of close"},
    # ── Acceleration Bands / Ulcer ────────────────────────────────────────────
    {"name": "ACCB_UPPER",      "category": "Volatility","col": "ACCB_UPPER",   "desc": "Acceleration Bands upper (20)"},
    {"name": "ACCB_MID",        "category": "Volatility","col": "ACCB_MID",     "desc": "Acceleration Bands middle"},
    {"name": "ACCB_LOWER",      "category": "Volatility","col": "ACCB_LOWER",   "desc": "Acceleration Bands lower"},
    {"name": "UI_14",           "category": "Volatility","col": "UI_14",        "desc": "Ulcer Index (14) — measures downside risk / drawdown"},
    # ── Choppiness / VHF ──────────────────────────────────────────────────────
    {"name": "CHOP_14",         "category": "Volatility","col": "CHOP_14",      "desc": "Choppiness Index (14): >61.8=choppy, <38.2=trending"},
    {"name": "VHF_28",          "category": "Volatility","col": "VHF_28",       "desc": "Vertical Horizontal Filter (28): high=trending, low=ranging"},

    # ══════════════════════════════════════════════════════════════════════════
    # VOLUME — flow, pressure, accumulation / distribution
    # ══════════════════════════════════════════════════════════════════════════

    {"name": "OBV",             "category": "Volume",  "col": "OBV",            "desc": "On-Balance Volume — cumulative buy/sell pressure"},
    {"name": "OBV_EMA_12",      "category": "Volume",  "col": "OBV_EMA_12",     "desc": "12-bar EMA of OBV"},
    {"name": "OBV_trend",       "category": "Volume",  "col": "OBV_trend",      "desc": "OBV vs EMA12: +1=rising, −1=falling"},
    {"name": "OBV_ZSCORE",      "category": "Volume",  "col": "OBV_ZSCORE",     "desc": "OBV normalised as 20-bar z-score"},
    {"name": "MFI_14",          "category": "Volume",  "col": "MFI_14",         "desc": "Money Flow Index (14) — volume-weighted RSI, 0–100"},
    {"name": "CMF_20",          "category": "Volume",  "col": "CMF_20",         "desc": "Chaikin Money Flow (20) — ±0.25 = strong flow"},
    {"name": "VWAP",            "category": "Volume",  "col": "VWAP",           "desc": "VWAP — rolling 14-bar volume-weighted average price"},
    {"name": "CLOSE_VS_VWAP",   "category": "Volume",  "col": "CLOSE_VS_VWAP",  "desc": "Distance from VWAP in % — above=bullish pressure"},
    {"name": "FI_13",           "category": "Volume",  "col": "FI_13",          "desc": "Force Index (13-EMA) — price×volume momentum"},
    {"name": "EOM_14",          "category": "Volume",  "col": "EOM_14",         "desc": "Ease of Movement (14) — price efficiency per unit volume"},
    {"name": "AD",              "category": "Volume",  "col": "AD",             "desc": "Chaikin A/D Line — cumulative accumulation/distribution"},
    {"name": "ADOSC",           "category": "Volume",  "col": "ADOSC",          "desc": "Chaikin A/D Oscillator (fast=3, slow=10)"},
    {"name": "PVT",             "category": "Volume",  "col": "PVT",            "desc": "Price Volume Trend — cumulative vol×pct_change"},
    {"name": "PVO",             "category": "Volume",  "col": "PVO",            "desc": "Percentage Volume Oscillator (EMA12−EMA26)/EMA26 ×100"},
    {"name": "PVO_signal",      "category": "Volume",  "col": "PVO_signal",     "desc": "PVO signal line (9-EMA)"},
    {"name": "VWMA_20",         "category": "Volume",  "col": "VWMA_20",        "desc": "Volume-Weighted MA (20) — price centre of gravity"},
    {"name": "VWMA_50",         "category": "Volume",  "col": "VWMA_50",        "desc": "Volume-Weighted MA (50)"},
    {"name": "NVI",             "category": "Volume",  "col": "NVI",            "desc": "Negative Volume Index — tracks smart money (low-volume days)"},
    {"name": "PVI",             "category": "Volume",  "col": "PVI",            "desc": "Positive Volume Index — tracks crowd money (high-volume days)"},
    {"name": "AR_14",           "category": "Volume",  "col": "AR_14",          "desc": "BRAR AR (14) — market sentiment via open/high/low ranges"},
    {"name": "BR_14",           "category": "Volume",  "col": "BR_14",          "desc": "BRAR BR (14) — buyer vs seller strength"},
    {"name": "BULL_POWER_13",   "category": "Volume",  "col": "BULL_POWER_13",  "desc": "Elder's Bull Power (13) — high − EMA13"},
    {"name": "BEAR_POWER_13",   "category": "Volume",  "col": "BEAR_POWER_13",  "desc": "Elder's Bear Power (13) — low − EMA13"},
    {"name": "KVO",             "category": "Volume",  "col": "KVO",            "desc": "Klinger Volume Oscillator (34,55) — volume trend"},
    {"name": "QSTICK_8",        "category": "Volume",  "col": "QSTICK_8",       "desc": "Q-Stick (8) — MA of candle body (close−open)"},
    {"name": "VOL_OSC",         "category": "Volume",  "col": "VOL_OSC",        "desc": "Volume Oscillator — (EMA5−EMA10)/EMA10 ×100"},
    {"name": "NET_VOL",         "category": "Volume",  "col": "NET_VOL",        "desc": "Net Volume — volume × sign(close−open)"},
    {"name": "volume_ratio",    "category": "Volume",  "col": "volume_ratio",   "desc": "Volume / 20-bar MA — >2 = volume spike"},
    {"name": "volume_spike",    "category": "Volume",  "col": "volume_spike",   "desc": "1 if volume_ratio > 2.0 (exceptional volume)"},

    # ══════════════════════════════════════════════════════════════════════════
    # PRICE DERIVED — calculated directly from OHLC, no smoothing
    # ══════════════════════════════════════════════════════════════════════════

    {"name": "TYPPRICE",        "category": "Price Derived","col": "TYPPRICE",   "desc": "Typical Price = (H+L+C) / 3"},
    {"name": "MEDPRICE",        "category": "Price Derived","col": "MEDPRICE",   "desc": "Median Price = (H+L) / 2"},
    {"name": "WCLPRICE",        "category": "Price Derived","col": "WCLPRICE",   "desc": "Weighted Close = (H+L+2C) / 4"},
    {"name": "AVGPRICE",        "category": "Price Derived","col": "AVGPRICE",   "desc": "Average Price = (O+H+L+C) / 4"},
    {"name": "BOP",             "category": "Price Derived","col": "BOP",        "desc": "Balance of Power = (C−O) / (H−L)"},
    {"name": "MIDPOINT_14",     "category": "Price Derived","col": "MIDPOINT_14","desc": "Mid-point of close over 14 bars"},
    {"name": "MIDPRICE_14",     "category": "Price Derived","col": "MIDPRICE_14","desc": "Mid-point of high/low over 14 bars"},
    {"name": "MAX_14",          "category": "Price Derived","col": "MAX_14",     "desc": "Highest close over 14 bars"},
    {"name": "MIN_14",          "category": "Price Derived","col": "MIN_14",     "desc": "Lowest close over 14 bars"},
    {"name": "SUM_14",          "category": "Price Derived","col": "SUM_14",     "desc": "14-bar sum of close"},
    # ── Linear Regression ──────────────────────────────────────────────────────
    {"name": "LINREG_14",           "category": "Price Derived","col": "LINREG_14",           "desc": "Linear Regression fitted value over 14 bars"},
    {"name": "LINREG_SLOPE_14",     "category": "Price Derived","col": "LINREG_SLOPE_14",     "desc": "Linear Regression slope (14) — rate of directional change"},
    {"name": "LINREG_SLOPE_5",      "category": "Price Derived","col": "LINREG_SLOPE_5",      "desc": "Linear Regression slope (5) — very fast"},
    {"name": "LINREG_INTERCEPT_14", "category": "Price Derived","col": "LINREG_INTERCEPT_14", "desc": "Linear Regression intercept (14)"},
    {"name": "LINREG_ANGLE_14",     "category": "Price Derived","col": "LINREG_ANGLE_14",     "desc": "Linear Regression angle in degrees (14)"},
    # ── Pivot points ───────────────────────────────────────────────────────────
    {"name": "PP",              "category": "Price Derived","col": "PP",         "desc": "Pivot Point = (20-bar H+L+avg_C) / 3"},
    {"name": "R1",              "category": "Price Derived","col": "R1",         "desc": "Resistance 1 = 2×PP − 20-bar low"},
    {"name": "R2",              "category": "Price Derived","col": "R2",         "desc": "Resistance 2 = PP + (20-bar H−L)"},
    {"name": "S1",              "category": "Price Derived","col": "S1",         "desc": "Support 1 = 2×PP − 20-bar high"},
    {"name": "S2",              "category": "Price Derived","col": "S2",         "desc": "Support 2 = PP − (20-bar H−L)"},
    {"name": "PP_dist_pct",     "category": "Price Derived","col": "PP_dist_pct","desc": "Distance from PP in % — positive=above pivot"},

    # ══════════════════════════════════════════════════════════════════════════
    # STATISTICS — rolling distributional measures
    # ══════════════════════════════════════════════════════════════════════════

    {"name": "STDDEV_10",       "category": "Statistics","col": "STDDEV_10",    "desc": "Rolling standard deviation of close (10)"},
    {"name": "STDDEV_20",       "category": "Statistics","col": "STDDEV_20",    "desc": "Rolling standard deviation of close (20)"},
    {"name": "STDDEV_50",       "category": "Statistics","col": "STDDEV_50",    "desc": "Rolling standard deviation of close (50)"},
    {"name": "VAR_10",          "category": "Statistics","col": "VAR_10",       "desc": "Rolling variance of close (10)"},
    {"name": "VAR_20",          "category": "Statistics","col": "VAR_20",       "desc": "Rolling variance of close (20)"},
    {"name": "VAR_50",          "category": "Statistics","col": "VAR_50",       "desc": "Rolling variance of close (50)"},
    {"name": "ZSCORE_10",       "category": "Statistics","col": "ZSCORE_10",    "desc": "Z-score of close over 10 bars: (close−mean)/std"},
    {"name": "ZSCORE_20",       "category": "Statistics","col": "ZSCORE_20",    "desc": "Z-score of close over 20 bars"},
    {"name": "ZSCORE_50",       "category": "Statistics","col": "ZSCORE_50",    "desc": "Z-score of close over 50 bars"},
    {"name": "PCT_RANK_10",     "category": "Statistics","col": "PCT_RANK_10",  "desc": "Percentile rank of current close in last 10 bars (0–1)"},
    {"name": "PCT_RANK_20",     "category": "Statistics","col": "PCT_RANK_20",  "desc": "Percentile rank in last 20 bars"},
    {"name": "PCT_RANK_50",     "category": "Statistics","col": "PCT_RANK_50",  "desc": "Percentile rank in last 50 bars"},
    {"name": "SKEW_20",         "category": "Statistics","col": "SKEW_20",      "desc": "Rolling skewness of close (20) — asymmetry of distribution"},
    {"name": "KURT_20",         "category": "Statistics","col": "KURT_20",      "desc": "Rolling kurtosis of close (20) — tail heaviness"},
    {"name": "MAD_20",          "category": "Statistics","col": "MAD_20",       "desc": "Mean Absolute Deviation of close (20)"},
    {"name": "MEDIAN_20",       "category": "Statistics","col": "MEDIAN_20",    "desc": "Rolling median of close (20)"},
    {"name": "QUANTILE_25_20",  "category": "Statistics","col": "QUANTILE_25_20","desc": "25th percentile of close over 20 bars"},
    {"name": "QUANTILE_75_20",  "category": "Statistics","col": "QUANTILE_75_20","desc": "75th percentile of close over 20 bars"},
    {"name": "AUTOCORR_1",      "category": "Statistics","col": "AUTOCORR_1",   "desc": "Autocorrelation lag-1 over 20-bar window — persistence test"},
    {"name": "AUTOCORR_2",      "category": "Statistics","col": "AUTOCORR_2",   "desc": "Autocorrelation lag-2 over 20-bar window"},
    {"name": "CORREL_CV_14",    "category": "Statistics","col": "CORREL_CV_14", "desc": "14-bar Pearson correlation of close vs volume"},
    {"name": "CORREL_HL_14",    "category": "Statistics","col": "CORREL_HL_14", "desc": "14-bar Pearson correlation of high vs low"},

    # ══════════════════════════════════════════════════════════════════════════
    # STRUCTURE — price action, market structure, swing levels
    # ══════════════════════════════════════════════════════════════════════════

    {"name": "candle_body",     "category": "Structure","col": "candle_body",   "desc": "Absolute candle body size in price units"},
    {"name": "candle_range",    "category": "Structure","col": "candle_range",  "desc": "Full candle range (H−L)"},
    {"name": "body_pct",        "category": "Structure","col": "body_pct",      "desc": "Body / range — 1=marubozu, 0=doji"},
    {"name": "upper_wick_pct",  "category": "Structure","col": "upper_wick_pct","desc": "Upper wick as fraction of full range"},
    {"name": "lower_wick_pct",  "category": "Structure","col": "lower_wick_pct","desc": "Lower wick as fraction of full range"},
    {"name": "PRICE_RANGE_PCT", "category": "Structure","col": "PRICE_RANGE_PCT","desc": "Price position within 20-bar H/L range: 0=at low, 100=at high"},
    {"name": "HH",              "category": "Structure","col": "HH",            "desc": "1 if current bar makes a new 20-bar high"},
    {"name": "LL",              "category": "Structure","col": "LL",            "desc": "1 if current bar makes a new 20-bar low"},
    {"name": "BARS_SINCE_HIGH", "category": "Structure","col": "BARS_SINCE_HIGH","desc": "Bars elapsed since the most recent all-time high"},
    {"name": "BARS_SINCE_LOW",  "category": "Structure","col": "BARS_SINCE_LOW", "desc": "Bars elapsed since the most recent all-time low"},
    {"name": "FRACTAL_BULL",    "category": "Structure","col": "FRACTAL_BULL",  "desc": "1 if bar is a Williams bullish fractal (5-bar lowest low)"},
    {"name": "FRACTAL_BEAR",    "category": "Structure","col": "FRACTAL_BEAR",  "desc": "1 if bar is a Williams bearish fractal (5-bar highest high)"},

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERNS — candlestick formations
    # ══════════════════════════════════════════════════════════════════════════

    {"name": "CDL_DOJI",            "category": "Patterns","col": "CDL_DOJI",           "desc": "Doji — body <10% of range; indecision (100/0)"},
    {"name": "CDL_SPINNING_TOP",    "category": "Patterns","col": "CDL_SPINNING_TOP",   "desc": "Spinning Top — small body (10–35%) with equal wicks"},
    {"name": "CDL_DRAGONFLY",       "category": "Patterns","col": "CDL_DRAGONFLY",      "desc": "Dragonfly Doji — lower wick >70% of range (bullish)"},
    {"name": "CDL_GRAVESTONE",      "category": "Patterns","col": "CDL_GRAVESTONE",     "desc": "Gravestone Doji — upper wick >70% of range (bearish)"},
    {"name": "CDL_BULL_MARUBOZU",   "category": "Patterns","col": "CDL_BULL_MARUBOZU",  "desc": "Bullish Marubozu — body >90% of range, bull"},
    {"name": "CDL_BEAR_MARUBOZU",   "category": "Patterns","col": "CDL_BEAR_MARUBOZU",  "desc": "Bearish Marubozu — body >90% of range, bear"},
    {"name": "CDL_HAMMER",          "category": "Patterns","col": "CDL_HAMMER",         "desc": "Hammer — lower wick >2× body, upper <50% body"},
    {"name": "CDL_INV_HAMMER",      "category": "Patterns","col": "CDL_INV_HAMMER",     "desc": "Inverted Hammer — upper wick >2× body"},
    {"name": "CDL_HANGING_MAN",     "category": "Patterns","col": "CDL_HANGING_MAN",    "desc": "Hanging Man — hammer shape at top (bearish)"},
    {"name": "CDL_SHOOTING_STAR",   "category": "Patterns","col": "CDL_SHOOTING_STAR",  "desc": "Shooting Star — upper wick >2× body at top (bearish)"},
    {"name": "CDL_BULL_ENGULFING",  "category": "Patterns","col": "CDL_BULL_ENGULFING", "desc": "Bullish Engulfing — bull candle fully covers prior bear"},
    {"name": "CDL_BEAR_ENGULFING",  "category": "Patterns","col": "CDL_BEAR_ENGULFING", "desc": "Bearish Engulfing — bear candle fully covers prior bull"},
    {"name": "CDL_BULL_HARAMI",     "category": "Patterns","col": "CDL_BULL_HARAMI",    "desc": "Bullish Harami — small bull inside large prior bear"},
    {"name": "CDL_BEAR_HARAMI",     "category": "Patterns","col": "CDL_BEAR_HARAMI",    "desc": "Bearish Harami — small bear inside large prior bull"},
    {"name": "CDL_DARK_CLOUD",      "category": "Patterns","col": "CDL_DARK_CLOUD",     "desc": "Dark Cloud Cover — bearish reversal two-candle pattern"},
    {"name": "CDL_PIERCING",        "category": "Patterns","col": "CDL_PIERCING",       "desc": "Piercing Line — bullish reversal two-candle pattern"},
    {"name": "CDL_TWEEZER_TOP",     "category": "Patterns","col": "CDL_TWEEZER_TOP",    "desc": "Tweezer Top — two candles sharing the same high"},
    {"name": "CDL_TWEEZER_BOTTOM",  "category": "Patterns","col": "CDL_TWEEZER_BOTTOM", "desc": "Tweezer Bottom — two candles sharing the same low"},
    {"name": "CDL_INSIDE_BAR",      "category": "Patterns","col": "CDL_INSIDE_BAR",     "desc": "Inside Bar — full range within prior bar (consolidation)"},
    {"name": "CDL_OUTSIDE_BAR",     "category": "Patterns","col": "CDL_OUTSIDE_BAR",    "desc": "Outside Bar — range engulfs prior bar (expansion)"},
    {"name": "CDL_MORNING_STAR",    "category": "Patterns","col": "CDL_MORNING_STAR",   "desc": "Morning Star — bullish three-candle reversal"},
    {"name": "CDL_EVENING_STAR",    "category": "Patterns","col": "CDL_EVENING_STAR",   "desc": "Evening Star — bearish three-candle reversal"},
    {"name": "CDL_3_WHITE_SOLDIERS","category": "Patterns","col": "CDL_3_WHITE_SOLDIERS","desc": "Three White Soldiers — strong bullish continuation"},
    {"name": "CDL_3_BLACK_CROWS",   "category": "Patterns","col": "CDL_3_BLACK_CROWS",  "desc": "Three Black Crows — strong bearish continuation"},
    {"name": "CDL_3_INSIDE_UP",     "category": "Patterns","col": "CDL_3_INSIDE_UP",    "desc": "Three Inside Up — bullish reversal confirmation"},
    {"name": "CDL_3_INSIDE_DOWN",   "category": "Patterns","col": "CDL_3_INSIDE_DOWN",  "desc": "Three Inside Down — bearish reversal confirmation"},

    # ══════════════════════════════════════════════════════════════════════════
    # REGIME — market state classification
    # ══════════════════════════════════════════════════════════════════════════

    {"name": "HIGH_VOL_REGIME", "category": "Regime",  "col": "HIGH_VOL_REGIME","desc": "1 if ATR14 is in the top quartile of last 100 bars"},
    {"name": "EMA50_slope",     "category": "Regime",  "col": "EMA50_slope",    "desc": "EMA50 slope as % over 5 bars — +ve=uptrend, −ve=downtrend"},
    {"name": "ema_aligned_bull","category": "Regime",  "col": "ema_aligned_bull","desc": "1 if EMA20 > EMA50 > EMA200 (full bull alignment)"},
    {"name": "CHOP_14",         "category": "Regime",  "col": "CHOP_14",        "desc": "Choppiness Index: >61.8=ranging, <38.2=trending"},
    {"name": "VHF_28",          "category": "Regime",  "col": "VHF_28",         "desc": "Vertical Horizontal Filter: high=trending, low=ranging"},
    {"name": "AMAT_fast",       "category": "Regime",  "col": "AMAT_fast",      "desc": "1 if fast EMA (8) > slow EMA (21)"},

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-TF — higher timeframe indicators injected onto primary bars
    # ══════════════════════════════════════════════════════════════════════════

    {"name": "HTF_{tf}_RSI_14",       "category": "Multi-TF","col": "HTF_{tf}_RSI_14",       "desc": "RSI(14) on higher TF — HTF overbought/oversold context"},
    {"name": "HTF_{tf}_EMA_20",       "category": "Multi-TF","col": "HTF_{tf}_EMA_20",       "desc": "EMA(20) on higher TF"},
    {"name": "HTF_{tf}_EMA_50",       "category": "Multi-TF","col": "HTF_{tf}_EMA_50",       "desc": "EMA(50) on higher TF — HTF trend level"},
    {"name": "HTF_{tf}_EMA_200",      "category": "Multi-TF","col": "HTF_{tf}_EMA_200",      "desc": "EMA(200) on higher TF — major HTF structure"},
    {"name": "HTF_{tf}_MACD_hist",    "category": "Multi-TF","col": "HTF_{tf}_MACD_hist",    "desc": "MACD histogram on higher TF — HTF momentum direction"},
    {"name": "HTF_{tf}_ADX_14",       "category": "Multi-TF","col": "HTF_{tf}_ADX_14",       "desc": "ADX(14) on higher TF — HTF trend strength"},
    {"name": "HTF_{tf}_BB_pct_20",    "category": "Multi-TF","col": "HTF_{tf}_BB_pct_20",    "desc": "Bollinger %B on higher TF — HTF price position in bands"},
    {"name": "HTF_{tf}_ATR_14",       "category": "Multi-TF","col": "HTF_{tf}_ATR_14",       "desc": "ATR(14) on higher TF — HTF volatility context"},
    {"name": "HTF_{tf}_SUPERT_dir",   "category": "Multi-TF","col": "HTF_{tf}_SUPERT_dir",   "desc": "Supertrend direction on higher TF (+1=bull, −1=bear)"},
    {"name": "HTF_{tf}_STOCH_K",      "category": "Multi-TF","col": "HTF_{tf}_STOCH_K",      "desc": "Stochastic %K on higher TF"},
    {"name": "HTF_{tf}_CCI_20",       "category": "Multi-TF","col": "HTF_{tf}_CCI_20",       "desc": "CCI(20) on higher TF"},
    {"name": "HTF_{tf}_MFI_14",       "category": "Multi-TF","col": "HTF_{tf}_MFI_14",       "desc": "Money Flow Index on higher TF"},
    {"name": "HTF_{tf}_volume_ratio", "category": "Multi-TF","col": "HTF_{tf}_volume_ratio", "desc": "Volume ratio on higher TF — HTF volume activity"},
    {"name": "HTF_{tf}_EMA50_slope",  "category": "Multi-TF","col": "HTF_{tf}_EMA50_slope",  "desc": "EMA50 slope on higher TF — quantifies HTF trend speed"},
    {"name": "HTF_{tf}_NATR_14",      "category": "Multi-TF","col": "HTF_{tf}_NATR_14",      "desc": "Normalised ATR on higher TF — relative volatility"},
    {"name": "HTF_{tf}_trend_dir",    "category": "Multi-TF","col": "HTF_{tf}_trend_dir",    "desc": "Derived: +1 if HTF close > HTF EMA50, −1 otherwise"},
    {"name": "HTF_{tf}_regime",       "category": "Multi-TF","col": "HTF_{tf}_regime",       "desc": "Derived: +1=trending_up, −1=trending_down, 0=ranging on HTF"},
    {"name": "HTF_{tf}_ema_20_50_cross","category":"Multi-TF","col":"HTF_{tf}_ema_20_50_cross","desc":"EMA20/50 cross on higher TF (+1=golden, −1=death)"},
    {"name": "HTF_{tf}_HIGH_VOL_REGIME","category":"Multi-TF","col":"HTF_{tf}_HIGH_VOL_REGIME","desc":"Volatility regime flag on higher TF"},
    {"name": "HTF_{tf}_PRICE_RANGE_PCT","category":"Multi-TF","col":"HTF_{tf}_PRICE_RANGE_PCT","desc":"Price position in 20-bar H/L range on higher TF"},
]

CATEGORIES = [
    "Trend", "Momentum", "Volatility", "Volume",
    "Price Derived", "Statistics", "Structure", "Patterns", "Regime", "Multi-TF",
]

# Example timeframes for documentation
EXAMPLE_TFS = ["1h", "4h", "1d"]


@bp.route("/")
def library_view():
    by_category = {cat: [] for cat in CATEGORIES}
    for ind in INDICATOR_CATALOG:
        cat = ind["category"]
        if cat in by_category:
            by_category[cat].append(ind)
    return render_template("indicators_lib/library.html",
                           by_category=by_category,
                           categories=CATEGORIES,
                           total=len(INDICATOR_CATALOG),
                           example_tfs=EXAMPLE_TFS)
