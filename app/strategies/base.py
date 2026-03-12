"""
BaseStrategy: adds every major technical analysis indicator to a DataFrame.
All custom strategies inherit from this and override populate_entry_signal()
and populate_exit_signal().

Uses the `ta` library (https://github.com/bukosabino/ta) which is pure Python
and installs cleanly on all platforms including Windows + Python 3.11.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseStrategy:
    """
    Base class for all CryptoAlgoFinder strategies.
    Provides populate_indicators() which adds 120+ indicators via the `ta` library
    plus manually-computed indicators (Supertrend, HMA, Fisher, CMO, Pivots, etc.)
    """

    name: str = "BaseStrategy"
    params: dict = {}

    def __init__(self, params: dict | None = None):
        if params:
            self.params = {**self.__class__.params, **params}

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators then compute entry/exit signals. Returns enriched df."""
        df = self.populate_indicators(df.copy())
        df = self.populate_entry_signal(df)
        df = self.populate_exit_signal(df)
        return df

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the full suite of technical indicators using the `ta` library
        plus manually-computed indicators.
        """
        if len(df) < 50:
            return df

        try:
            import ta.trend as tr
            import ta.momentum as mom
            import ta.volatility as vol_mod
            import ta.volume as volm
        except ImportError:
            logger.warning("ta library not installed. Run: pip install ta==0.11.0")
            return df

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]
        open_ = df["open"]

        try:
            # ── Trend ─────────────────────────────────────────────────────────
            for period in [8, 13, 20, 21, 50, 100, 200]:
                df[f"EMA_{period}"] = tr.EMAIndicator(close, window=period, fillna=False).ema_indicator()
                df[f"SMA_{period}"] = tr.SMAIndicator(close, window=period, fillna=False).sma_indicator()
                df[f"WMA_{period}"] = tr.WMAIndicator(close, window=period, fillna=False).wma()

            # TEMA and DEMA (triple/double exponential MA, faster & less lag)
            for period in [9, 21, 50]:
                try:
                    df[f"TEMA_{period}"] = tr.TEMAIndicator(close, window=period, fillna=False).tema()
                    df[f"DEMA_{period}"] = tr.DEMAIndicator(close, window=period, fillna=False).dema()
                except Exception:
                    pass

            # HMA (Hull Moving Average) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
            for period in [9, 20, 50]:
                df[f"HMA_{period}"] = _hull_moving_average(close, period)

            # MACD
            macd_obj = tr.MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
            df["MACD"]        = macd_obj.macd()
            df["MACD_signal"] = macd_obj.macd_signal()
            df["MACD_hist"]   = macd_obj.macd_diff()

            # ADX + DMI
            adx_obj = tr.ADXIndicator(high, low, close, window=14, fillna=False)
            df["ADX_14"] = adx_obj.adx()
            df["DMP_14"] = adx_obj.adx_pos()
            df["DMN_14"] = adx_obj.adx_neg()

            # CCI (Commodity Channel Index) — oversold < -100, overbought > +100
            df["CCI_20"] = tr.CCIIndicator(high, low, close, window=20, constant=0.015, fillna=False).cci()

            # Vortex Indicator — trend direction
            try:
                vortex = tr.VortexIndicator(high, low, close, window=14, fillna=False)
                df["VI_pos"] = vortex.vortex_indicator_pos()
                df["VI_neg"] = vortex.vortex_indicator_neg()
            except Exception:
                pass

            # TRIX — triple smoothed EMA momentum
            try:
                df["TRIX_15"] = tr.TRIXIndicator(close, window=15, fillna=False).trix()
            except Exception:
                pass

            # DPO (Detrended Price Oscillator) — separates short cycles
            try:
                df["DPO_20"] = tr.DPOIndicator(close, window=20, fillna=False).dpo()
            except Exception:
                pass

            # KST (Know Sure Thing) — long-term momentum
            try:
                kst_obj = tr.KSTIndicator(close, fillna=False)
                df["KST"]        = kst_obj.kst()
                df["KST_signal"] = kst_obj.kst_sig()
            except Exception:
                pass

            # Mass Index — reversal signal in high-volatility compression
            try:
                df["MASS_INDEX"] = tr.MassIndex(high, low, window_fast=9, window_slow=25, fillna=False).mass_index()
            except Exception:
                pass

            # Ichimoku
            ichi = tr.IchimokuIndicator(high, low, window1=9, window2=26, window3=52, fillna=False)
            df["ICH_tenkan"]   = ichi.ichimoku_conversion_line()
            df["ICH_kijun"]    = ichi.ichimoku_base_line()
            df["ICH_senkou_a"] = ichi.ichimoku_a()
            df["ICH_senkou_b"] = ichi.ichimoku_b()
            # Cloud position: price vs cloud
            df["ICH_above_cloud"] = ((close.values > df["ICH_senkou_a"].values) & (close.values > df["ICH_senkou_b"].values)).astype(int)
            df["ICH_below_cloud"] = ((close.values < df["ICH_senkou_a"].values) & (close.values < df["ICH_senkou_b"].values)).astype(int)

            # Aroon
            aroon = tr.AroonIndicator(high, low, window=25, fillna=False)
            df["AROON_up"]   = aroon.aroon_up()
            df["AROON_down"] = aroon.aroon_down()
            df["AROON_osc"]  = df["AROON_up"] - df["AROON_down"]

            # PSAR (Parabolic SAR)
            psar_obj = tr.PSARIndicator(high, low, close, step=0.02, max_step=0.2, fillna=False)
            df["PSAR_up"]   = psar_obj.psar_up()
            df["PSAR_down"] = psar_obj.psar_down()
            # PSAR direction: +1 = bullish (price above PSAR), -1 = bearish
            psar_val = psar_obj.psar()
            df["PSAR_dir"] = ((close.values > psar_val.values).astype(int) * 2 - 1)

            # ── Supertrend (manual — ATR-based trend filter) ──────────────────
            for period, mult in [(10, 3.0), (14, 2.0)]:
                supert, supert_dir = _supertrend(high, low, close, period, mult)
                df[f"SUPERT_{period}_{int(mult)}"] = supert
                df[f"SUPERT_dir_{period}_{int(mult)}"] = supert_dir
            # Primary Supertrend (10,3) as SUPERT_dir for backwards compat
            df["SUPERT_dir"] = df["SUPERT_dir_10_3"]

            # ── Momentum ──────────────────────────────────────────────────────
            for period in [7, 14, 21]:
                df[f"RSI_{period}"] = mom.RSIIndicator(close, window=period, fillna=False).rsi()

            # Stochastic
            stoch_obj = mom.StochasticOscillator(high, low, close, window=14, smooth_window=3, fillna=False)
            df["STOCH_K"] = stoch_obj.stoch()
            df["STOCH_D"] = stoch_obj.stoch_signal()

            # Stochastic RSI
            stochrsi_obj = mom.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3, fillna=False)
            df["STOCHRSI_K"] = stochrsi_obj.stochrsi_k()
            df["STOCHRSI_D"] = stochrsi_obj.stochrsi_d()

            # Williams %R
            df["WILLR_14"] = mom.WilliamsRIndicator(high, low, close, lbp=14, fillna=False).williams_r()

            # ROC
            df["ROC_10"] = mom.ROCIndicator(close, window=10, fillna=False).roc()
            df["ROC_20"] = mom.ROCIndicator(close, window=20, fillna=False).roc()

            # Awesome Oscillator
            df["AO"] = mom.AwesomeOscillatorIndicator(high, low, window1=5, window2=34, fillna=False).awesome_oscillator()

            # KAMA
            df["KAMA"] = mom.KAMAIndicator(close, window=10, pow1=2, pow2=30, fillna=False).kama()

            # CMO (Chande Momentum Oscillator) — manual implementation
            df["CMO_14"] = _cmo(close, 14)

            # Fisher Transform — converts prices to Gaussian normal distribution
            df["FISHER"] = _fisher_transform(high, low, period=9)

            # Ultimate Oscillator
            try:
                df["UO"] = mom.UltimateOscillator(high, low, close, window1=7, window2=14, window3=28,
                                                    weight1=4.0, weight2=2.0, weight3=1.0, fillna=False).ultimate_oscillator()
            except Exception:
                pass

            # PPO (Percentage Price Oscillator) — like MACD but normalized
            try:
                ppo = mom.PercentagePriceOscillator(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
                df["PPO"]        = ppo.ppo()
                df["PPO_signal"] = ppo.ppo_signal()
                df["PPO_hist"]   = ppo.ppo_hist()
            except Exception:
                pass

            # ── Volatility ────────────────────────────────────────────────────
            for period in [14, 20]:
                bb = vol_mod.BollingerBands(close, window=period, window_dev=2, fillna=False)
                df[f"BB_upper_{period}"] = bb.bollinger_hband()
                df[f"BB_mid_{period}"]   = bb.bollinger_mavg()
                df[f"BB_lower_{period}"] = bb.bollinger_lband()
                df[f"BB_width_{period}"] = bb.bollinger_wband()
                df[f"BB_pct_{period}"]   = bb.bollinger_pband()

            for period in [7, 14, 21]:
                df[f"ATR_{period}"] = vol_mod.AverageTrueRange(high, low, close, window=period, fillna=False).average_true_range()

            kc = vol_mod.KeltnerChannel(high, low, close, window=20, window_atr=10, fillna=False)
            df["KC_upper"]  = kc.keltner_channel_hband()
            df["KC_lower"]  = kc.keltner_channel_lband()
            df["KC_middle"] = kc.keltner_channel_mband()
            # Squeeze: BB inside KC (Bollinger Squeeze)
            df["BB_SQUEEZE"] = (
                (df["BB_upper_20"] < df["KC_upper"]) & (df["BB_lower_20"] > df["KC_lower"])
            ).astype(int)

            dc = vol_mod.DonchianChannel(high, low, close, window=20, fillna=False)
            df["DC_upper"]  = dc.donchian_channel_hband()
            df["DC_lower"]  = dc.donchian_channel_lband()
            df["DC_middle"] = dc.donchian_channel_mband()

            # Normalized ATR (ATR as % of price — comparable across pairs/times)
            df["NATR_14"] = (df["ATR_14"] / close) * 100

            # Historical Volatility (20-bar rolling std of log returns, annualized)
            log_ret = np.log(close / close.shift(1))
            df["HV_20"] = log_ret.rolling(20).std() * np.sqrt(365 * 24)  # hourly bars

            # ── Volume ────────────────────────────────────────────────────────
            df["OBV"]    = volm.OnBalanceVolumeIndicator(close, vol, fillna=False).on_balance_volume()
            df["MFI_14"] = volm.MFIIndicator(high, low, close, vol, window=14, fillna=False).money_flow_index()
            df["CMF_20"] = volm.ChaikinMoneyFlowIndicator(high, low, close, vol, window=20, fillna=False).chaikin_money_flow()
            df["VWAP"]   = volm.VolumeWeightedAveragePrice(high, low, close, vol, window=14, fillna=False).volume_weighted_average_price()

            # Force Index
            try:
                df["FI_13"] = volm.ForceIndexIndicator(close, vol, window=13, fillna=False).force_index()
            except Exception:
                pass

            # Ease of Movement
            try:
                df["EOM_14"] = volm.EaseOfMovementIndicator(high, low, vol, window=14, fillna=False).ease_of_movement()
            except Exception:
                pass

            # OBV trend (OBV above/below its own EMA)
            df["OBV_EMA_12"] = df["OBV"].ewm(span=12, adjust=False).mean()
            df["OBV_trend"] = (df["OBV"] > df["OBV_EMA_12"]).astype(int) * 2 - 1  # +1 or -1

            # ── Derived / Price-Action ────────────────────────────────────────
            df["candle_body"]      = (close - open_).abs()
            df["candle_range"]     = high - low
            df["body_pct"]         = df["candle_body"] / df["candle_range"].replace(0, float("nan"))
            df["upper_wick"]       = high - df[["open", "close"]].max(axis=1)
            df["lower_wick"]       = df[["open", "close"]].min(axis=1) - low
            df["upper_wick_pct"]   = df["upper_wick"] / df["candle_range"].replace(0, float("nan"))
            df["lower_wick_pct"]   = df["lower_wick"] / df["candle_range"].replace(0, float("nan"))
            df["close_pct_change"] = close.pct_change() * 100

            # Volume ratio
            vol_ma = vol.rolling(20).mean()
            df["volume_ratio"] = vol / vol_ma.replace(0, float("nan"))
            df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

            # Price distance from key MAs (as % of price)
            for ma in [20, 50, 200]:
                df[f"close_vs_EMA_{ma}"] = (close / df[f"EMA_{ma}"] - 1) * 100

            # EMA cross signals: +1 = bullish cross, -1 = bearish cross, 0 = no cross
            df["ema_20_50_cross"] = (
                (df["EMA_20"] > df["EMA_50"]).astype(int) -
                (df["EMA_20"].shift(1) > df["EMA_50"].shift(1)).astype(int)
            )
            df["ema_50_200_cross"] = (
                (df["EMA_50"] > df["EMA_200"]).astype(int) -
                (df["EMA_50"].shift(1) > df["EMA_200"].shift(1)).astype(int)
            )
            # EMA alignment: all 3 EMAs aligned bullish
            df["ema_aligned_bull"] = (
                (df["EMA_20"] > df["EMA_50"]) & (df["EMA_50"] > df["EMA_200"])
            ).astype(int)

            # ── Pivot Points (previous 20-bar high/low based levels) ──────────
            df = _add_pivot_points(df, high, low, close)

            # ── Market Structure ──────────────────────────────────────────────
            df = _add_market_structure(df, high, low, close)

            # ── Candlestick patterns ──────────────────────────────────────────
            df = _add_candle_patterns(df)

        except Exception as e:
            logger.warning("populate_indicators error: %s", e, exc_info=True)

        # ── Extended indicator suite (200+ additional indicators) ──────────────
        _add_extended_indicators(df, open_, high, low, close, vol)

        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override in subclasses. Must add column 'entry_signal' (1=buy, 0=no).
        Default: EMA20 > EMA50 AND RSI14 40-70 AND MACD hist > 0 AND ADX > 20.
        """
        if "EMA_20" not in df.columns:
            df["entry_signal"] = 0
            return df
        df["entry_signal"] = (
            (df["EMA_20"] > df["EMA_50"]) &
            (df["RSI_14"] < 70) &
            (df["RSI_14"] > 40) &
            (df["MACD_hist"] > 0) &
            (df["ADX_14"] > 20)
        ).astype(int)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override in subclasses. Must add column 'exit_signal' (1=sell, 0=no).
        Default: RSI14 > 75 OR EMA20 crosses below EMA50.
        """
        if "RSI_14" not in df.columns:
            df["exit_signal"] = 0
            return df
        df["exit_signal"] = (
            (df["RSI_14"] > 75) |
            (df["ema_20_50_cross"] == -1)
        ).astype(int)
        return df


# ── Manual indicator implementations ─────────────────────────────────────────

def _hull_moving_average(close: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    half = max(1, period // 2)
    sqrt_p = max(2, int(np.sqrt(period)))
    wma_half = close.rolling(half).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    wma_full = close.rolling(period).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    raw = 2 * wma_half - wma_full
    return raw.rolling(sqrt_p).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator. Returns (supertrend_line, direction).
    direction: +1 = bullish (price above supertrend), -1 = bearish.

    Uses pre-allocated numpy arrays instead of pandas Series lookups —
    eliminates .iloc[], .get(), and index lookups for 5–10× speedup.
    """
    hl2 = (high + low) / 2
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Pre-extract to numpy — direct integer indexing, no pandas overhead
    n         = len(close)
    ub_arr    = upper_band.values
    lb_arr    = lower_band.values
    close_arr = close.values
    st_arr    = np.full(n, np.nan)
    dir_arr   = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        prev_st = st_arr[i - 1]
        c_prev  = close_arr[i - 1]
        c_curr  = close_arr[i]
        lb_i    = lb_arr[i]
        ub_i    = ub_arr[i]

        # Carry-forward band logic
        if np.isnan(prev_st):
            final_lb = lb_i
            final_ub = ub_i
        else:
            final_lb = lb_i if (lb_i > prev_st or c_prev < prev_st) else prev_st
            final_ub = ub_i if (ub_i < prev_st or c_prev > prev_st) else prev_st

        # Use direction flag instead of float equality (prev_st == final_ub)
        # to determine which band was being tracked — cleaner and avoids
        # any floating-point equality edge cases.
        if dir_arr[i - 1] <= 0:  # was bearish (tracking upper band) or uninitialized
            if c_curr > final_ub:
                st_arr[i]  = final_lb
                dir_arr[i] = 1
            else:
                st_arr[i]  = final_ub
                dir_arr[i] = -1
        else:                     # was bullish (tracking lower band)
            if c_curr < final_lb:
                st_arr[i]  = final_ub
                dir_arr[i] = -1
            else:
                st_arr[i]  = final_lb
                dir_arr[i] = 1

    return pd.Series(st_arr, index=close.index), pd.Series(dir_arr.astype(int), index=close.index)


def _cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator: (up_sum - down_sum) / (up_sum + down_sum) * 100"""
    delta = close.diff()
    up   = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    up_sum   = up.rolling(period).sum()
    down_sum = down.rolling(period).sum()
    total    = up_sum + down_sum
    return ((up_sum - down_sum) / total.replace(0, float("nan"))) * 100


def _fisher_transform(high: pd.Series, low: pd.Series, period: int = 9) -> pd.Series:
    """
    Fisher Transform: converts price into a Gaussian normal distribution.
    Extremes (-2.5 / +2.5) signal potential reversals.
    """
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    hlrange = (highest - lowest).replace(0, float("nan"))

    value = 2 * ((high + low) / 2 - lowest) / hlrange - 1
    value = value.clip(-0.999, 0.999)  # avoid log(0)

    fisher = 0.5 * np.log((1 + value) / (1 - value))
    return fisher.ffill()


def _add_pivot_points(df: pd.DataFrame,
                      high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Rolling pivot points based on the previous 20-bar range.
    PP = (H + L + C) / 3
    R1 = 2*PP - L, R2 = PP + (H - L)
    S1 = 2*PP - H, S2 = PP - (H - L)
    Also adds price distance from PP as percentage.
    """
    roll_h = high.rolling(20).max()
    roll_l = low.rolling(20).min()
    roll_c = close.rolling(20).mean()

    pp = (roll_h + roll_l + roll_c) / 3
    df["PP"]  = pp
    df["R1"]  = 2 * pp - roll_l
    df["R2"]  = pp + (roll_h - roll_l)
    df["S1"]  = 2 * pp - roll_h
    df["S2"]  = pp - (roll_h - roll_l)
    df["PP_dist_pct"] = (close / pp - 1) * 100

    return df


def _add_market_structure(df: pd.DataFrame,
                          high: pd.Series, low: pd.Series, close: pd.Series,
                          lookback: int = 20) -> pd.DataFrame:
    """
    Market structure detection:
    - HH/HL/LH/LL flags over a rolling window
    - Trend regime: 1 = uptrend, -1 = downtrend, 0 = ranging
    - Price position within recent range (0–100%)
    """
    roll_h = high.rolling(lookback).max()
    roll_l = low.rolling(lookback).min()
    roll_range = (roll_h - roll_l).replace(0, float("nan"))

    # Price position within recent range (0 = at low, 100 = at high)
    df["PRICE_RANGE_PCT"] = ((close - roll_l) / roll_range) * 100

    # Higher High: current bar high > previous lookback max
    df["HH"] = (high > high.shift(1).rolling(lookback).max()).astype(int)
    # Lower Low: current bar low < previous lookback min
    df["LL"] = (low < low.shift(1).rolling(lookback).min()).astype(int)

    # Trend regime based on EMA50 slope
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema50_slope = (ema50 - ema50.shift(5)) / ema50.shift(5) * 100
    df["EMA50_slope"] = ema50_slope

    # Volatility regime: is current ATR in the top quartile of recent ATR history?
    atr14 = df.get("ATR_14", high - low)
    atr_rolling_q75 = atr14.rolling(100).quantile(0.75)
    df["HIGH_VOL_REGIME"] = (atr14 > atr_rolling_q75).astype(int)

    # Bars since all-time high / low — fully vectorized (no Python loop)
    high_s = pd.Series(high.values)
    low_s  = pd.Series(low.values)

    running_max = high_s.expanding().max()
    is_new_high = (high_s >= running_max.shift(1).fillna(-np.inf)).astype(int)
    group_h     = is_new_high.cumsum()
    df["BARS_SINCE_HIGH"] = group_h.groupby(group_h).cumcount().astype(float).values

    running_min = low_s.expanding().min()
    is_new_low  = (low_s <= running_min.shift(1).fillna(np.inf)).astype(int)
    group_l     = is_new_low.cumsum()
    df["BARS_SINCE_LOW"]  = group_l.groupby(group_l).cumcount().astype(float).values

    return df


# ── Candlestick Patterns ──────────────────────────────────────────────────────

def _add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure price-action candlestick pattern detection (no TA-Lib required).
    Each column is +100 (bullish), -100 (bearish), or 0.
    Implements 25+ standard patterns.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body  = (c - o).abs()
    rng   = (h - l).replace(0, float("nan"))
    upper = h - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - l
    is_bull = c > o
    is_bear = c < o

    # ── Single-candle patterns ────────────────────────────────────────────────

    # Doji: body < 10% of range
    df["CDL_DOJI"] = ((body / rng) < 0.1).map({True: 100, False: 0})

    # Spinning Top: small body (10–30%) with significant wicks on both sides
    df["CDL_SPINNING_TOP"] = (
        ((body / rng) > 0.1) & ((body / rng) < 0.35) &
        (upper > body * 0.5) & (lower > body * 0.5)
    ).map({True: 100, False: 0})

    # Marubozu (strong trend candles: body > 90% of range, minimal wicks)
    df["CDL_BULL_MARUBOZU"] = (is_bull & ((body / rng) > 0.90)).map({True: 100, False: 0})
    df["CDL_BEAR_MARUBOZU"] = (is_bear & ((body / rng) > 0.90)).map({True: -100, False: 0})

    # Hammer: lower wick > 2× body, upper wick < 50% body, bullish
    df["CDL_HAMMER"] = (
        (lower > 2 * body) & (upper < body * 0.5) & is_bull
    ).map({True: 100, False: 0})

    # Inverted Hammer: upper wick > 2× body, lower wick < 50% body, bullish body (reversal after downtrend)
    df["CDL_INV_HAMMER"] = (
        (upper > 2 * body) & (lower < body * 0.5) & is_bull
    ).map({True: 100, False: 0})

    # Hanging Man: hammer shape but bearish context (same shape, different context)
    df["CDL_HANGING_MAN"] = (
        (lower > 2 * body) & (upper < body * 0.5) & is_bear
    ).map({True: -100, False: 0})

    # Shooting Star: upper wick > 2× body, lower wick < 50% body, bearish
    df["CDL_SHOOTING_STAR"] = (
        (upper > 2 * body) & (lower < body * 0.5) & is_bear
    ).map({True: -100, False: 0})

    # Dragonfly Doji: very small body + lower wick dominates
    df["CDL_DRAGONFLY"] = (
        ((body / rng) < 0.1) & (lower > rng * 0.7)
    ).map({True: 100, False: 0})

    # Gravestone Doji: very small body + upper wick dominates
    df["CDL_GRAVESTONE"] = (
        ((body / rng) < 0.1) & (upper > rng * 0.7)
    ).map({True: -100, False: 0})

    # ── Two-candle patterns ───────────────────────────────────────────────────
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_body = (prev_c - prev_o).abs()
    prev_bull  = prev_c > prev_o
    prev_bear  = prev_c < prev_o

    # Bullish Engulfing: prev red, current green, fully engulfs
    df["CDL_BULL_ENGULFING"] = (
        prev_bear & is_bull & (o <= prev_c) & (c >= prev_o)
    ).map({True: 100, False: 0})

    # Bearish Engulfing: prev green, current red, fully engulfs
    df["CDL_BEAR_ENGULFING"] = (
        prev_bull & is_bear & (o >= prev_c) & (c <= prev_o)
    ).map({True: -100, False: 0})

    # Bullish Harami: large red bar, small green body inside
    df["CDL_BULL_HARAMI"] = (
        prev_bear & is_bull &
        (o > prev_c) & (c < prev_o) &
        (body < prev_body * 0.5)
    ).map({True: 100, False: 0})

    # Bearish Harami: large green bar, small red body inside
    df["CDL_BEAR_HARAMI"] = (
        prev_bull & is_bear &
        (o < prev_c) & (c > prev_o) &
        (body < prev_body * 0.5)
    ).map({True: -100, False: 0})

    # Dark Cloud Cover: prev large bull, current opens above prev high, closes below midpoint
    df["CDL_DARK_CLOUD"] = (
        prev_bull & is_bear &
        (o > prev_h) &
        (c < (prev_o + prev_c) / 2) &
        (c > prev_o)
    ).map({True: -100, False: 0})

    # Piercing Line: prev large bear, current opens below prev low, closes above midpoint
    df["CDL_PIERCING"] = (
        prev_bear & is_bull &
        (o < prev_l) &
        (c > (prev_o + prev_c) / 2) &
        (c < prev_o)
    ).map({True: 100, False: 0})

    # Tweezer Top: two candles at same high (potential reversal)
    df["CDL_TWEEZER_TOP"] = (
        ((h - prev_h).abs() < rng * 0.02) & prev_bull & is_bear
    ).map({True: -100, False: 0})

    # Tweezer Bottom: two candles at same low (potential reversal)
    df["CDL_TWEEZER_BOTTOM"] = (
        ((l - prev_l).abs() < rng * 0.02) & prev_bear & is_bull
    ).map({True: 100, False: 0})

    # Inside Bar (consolidation): current range is entirely within prev range
    df["CDL_INSIDE_BAR"] = (
        (h <= prev_h) & (l >= prev_l)
    ).map({True: 100, False: 0})

    # Outside Bar (expansion): current range engulfs prev range
    df["CDL_OUTSIDE_BAR"] = (
        (h > prev_h) & (l < prev_l)
    ).map({True: 100, False: 0})

    # ── Three-candle patterns ─────────────────────────────────────────────────
    prev2_o = o.shift(2)
    prev2_c = c.shift(2)
    prev2_bull = prev2_c > prev2_o
    prev2_bear = prev2_c < prev2_o

    # Three White Soldiers: 3 consecutive bullish candles, each opening within prev body, closing higher
    df["CDL_3_WHITE_SOLDIERS"] = (
        is_bull & prev_bull & prev2_bull &
        (o > prev_o) & (o < prev_c) &
        (prev_o > prev2_o) & (prev_o < prev2_c) &
        (c > prev_c) & (prev_c > prev2_c)
    ).map({True: 100, False: 0})

    # Three Black Crows: 3 consecutive bearish candles
    df["CDL_3_BLACK_CROWS"] = (
        is_bear & prev_bear & prev2_bull &
        (o < prev_o) & (o > prev_c) &
        (c < prev_c) & (prev_c < prev2_c)
    ).map({True: -100, False: 0})

    # Morning Star: prev2 bear, prev doji/small, current bull — bullish reversal
    prev_small = prev_body < (body * 0.3)
    df["CDL_MORNING_STAR"] = (
        prev2_bear & prev_small & is_bull &
        (c > (prev2_o + prev2_c) / 2)
    ).map({True: 100, False: 0})

    # Evening Star: prev2 bull, prev doji/small, current bear — bearish reversal
    prev2_bull_body = (prev2_c - prev2_o)
    df["CDL_EVENING_STAR"] = (
        prev2_bull & prev_small & is_bear &
        (c < (prev2_o + prev2_c) / 2)
    ).map({True: -100, False: 0})

    # Three Inside Up: bearish → harami → bullish confirmation
    df["CDL_3_INSIDE_UP"] = (
        prev2_bear & prev_bull &
        (prev_o > prev2_c) & (prev_c < prev2_o) &
        is_bull & (c > prev_c)
    ).map({True: 100, False: 0})

    # Three Inside Down: bullish → harami → bearish confirmation
    df["CDL_3_INSIDE_DOWN"] = (
        prev2_bull & prev_bear &
        (prev_o < prev2_c) & (prev_c > prev2_o) &
        is_bear & (c < prev_c)
    ).map({True: -100, False: 0})

    return df


# ── Extended indicator suite ──────────────────────────────────────────────────

def _add_extended_indicators(df: pd.DataFrame,
                              open_: pd.Series, high: pd.Series,
                              low: pd.Series, close: pd.Series,
                              vol: pd.Series) -> None:
    """
    Adds 200+ indicators beyond the base `ta` suite. Modifies df in-place.
    Uses pandas_ta where available; falls back to pure pandas/numpy.
    Every call is wrapped in try/except — a single failure never aborts the rest.
    """
    if len(df) < 30:
        return

    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    def _add(col: str, val) -> None:
        """Assign value to df[col] only if the column does not already exist."""
        if val is None or col in df.columns:
            return
        if isinstance(val, pd.Series):
            df[col] = val.values
        elif isinstance(val, np.ndarray):
            df[col] = val
        else:
            df[col] = val

    def _add_df(result, prefix: str = "") -> None:
        """Add all columns of a DataFrame result, optionally with a prefix."""
        if result is None or not isinstance(result, pd.DataFrame):
            return
        for col in result.columns:
            dest = f"{prefix}{col}" if prefix else col
            _add(dest, result[col])

    # ── Price-derived (no library needed) ─────────────────────────────────────
    _add("TYPPRICE",  (high + low + close) / 3)
    _add("MEDPRICE",  (high + low) / 2)
    _add("WCLPRICE",  (high + low + 2 * close) / 4)
    _add("AVGPRICE",  (open_ + high + low + close) / 4)
    _add("TRANGE",    _safe(lambda: pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)))
    _add("BOP",       _safe(lambda: (close - open_) / (high - low).replace(0, float("nan"))))
    _add("MOM_10",    _safe(lambda: close.diff(10)))
    _add("MOM_20",    _safe(lambda: close.diff(20)))
    _add("ROCP_10",   _safe(lambda: close.pct_change(10)))
    _add("ROCR_10",   _safe(lambda: close / close.shift(10)))
    _add("ROCR100_10",_safe(lambda: (close / close.shift(10)) * 100))
    _add("MIDPOINT_14", _safe(lambda: (close.rolling(14).max() + close.rolling(14).min()) / 2))
    _add("MIDPRICE_14", _safe(lambda: (high.rolling(14).max() + low.rolling(14).min()) / 2))
    _add("MAX_14",    _safe(lambda: close.rolling(14).max()))
    _add("MIN_14",    _safe(lambda: close.rolling(14).min()))
    _add("SUM_14",    _safe(lambda: close.rolling(14).sum()))
    _add("NET_VOL",   _safe(lambda: vol * np.sign(close - open_)))
    _add("VWMA_20",   _safe(lambda: (close * vol).rolling(20).sum() / vol.rolling(20).sum()))
    _add("VWMA_50",   _safe(lambda: (close * vol).rolling(50).sum() / vol.rolling(50).sum()))

    # APO (Absolute Price Oscillator) — manual
    _add("APO_12_26", _safe(lambda: close.ewm(span=12, adjust=False).mean() -
                                    close.ewm(span=26, adjust=False).mean()))

    # BETA (rolling regression of close returns vs their own lagged MA)
    _add("BETA_5", _safe(lambda: close.pct_change().rolling(5).corr(
        close.pct_change().rolling(5).mean().rolling(5).apply(lambda x: x[-1], raw=True)
    )))

    # CORREL (rolling Pearson correlation close vs volume)
    _add("CORREL_CV_14", _safe(lambda: close.rolling(14).corr(vol)))
    _add("CORREL_HL_14", _safe(lambda: high.rolling(14).corr(low)))

    # ── Rolling statistics ─────────────────────────────────────────────────────
    for p in [10, 20, 50]:
        _add(f"STDDEV_{p}",  _safe(lambda p=p: close.rolling(p).std()))
        _add(f"VAR_{p}",     _safe(lambda p=p: close.rolling(p).var()))
        _add(f"ZSCORE_{p}",  _safe(lambda p=p: (close - close.rolling(p).mean()) /
                                                 close.rolling(p).std().replace(0, float("nan"))))
        _add(f"PCT_RANK_{p}", _safe(lambda p=p: close.rolling(p).rank(pct=True)))
    _add("SKEW_20",   _safe(lambda: close.rolling(20).skew()))
    _add("KURT_20",   _safe(lambda: close.rolling(20).kurt()))
    _add("MAD_20",    _safe(lambda: close.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)))
    _add("AUTOCORR_1",_safe(lambda: close.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)))
    _add("AUTOCORR_2",_safe(lambda: close.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=2), raw=False)))
    _add("MEDIAN_20", _safe(lambda: close.rolling(20).median()))
    _add("QUANTILE_75_20", _safe(lambda: close.rolling(20).quantile(0.75)))
    _add("QUANTILE_25_20", _safe(lambda: close.rolling(20).quantile(0.25)))

    # Linear Regression family (pure numpy rolling polyfit)
    def _linreg(series: pd.Series, window: int, mode: str = "value") -> pd.Series:
        """Rolling linear regression. mode: value | slope | intercept | angle | tsf"""
        arr = series.values.astype(float)
        out = np.full(len(arr), np.nan)
        x = np.arange(window, dtype=float)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1: i + 1]
            if np.any(np.isnan(y)):
                continue
            slope, intercept = np.polyfit(x, y, 1)
            if mode == "slope":
                out[i] = slope
            elif mode == "intercept":
                out[i] = intercept
            elif mode == "angle":
                out[i] = np.degrees(np.arctan(slope))
            elif mode == "tsf":
                out[i] = slope * window + intercept   # one-bar-ahead forecast
            else:
                out[i] = slope * (window - 1) + intercept  # current fitted value
        return pd.Series(out, index=series.index)

    _add("LINREG_14",           _safe(lambda: _linreg(close, 14, "value")))
    _add("LINREG_SLOPE_14",     _safe(lambda: _linreg(close, 14, "slope")))
    _add("LINREG_INTERCEPT_14", _safe(lambda: _linreg(close, 14, "intercept")))
    _add("LINREG_ANGLE_14",     _safe(lambda: _linreg(close, 14, "angle")))
    _add("TSF_14",              _safe(lambda: _linreg(close, 14, "tsf")))
    _add("LINREG_SLOPE_5",      _safe(lambda: _linreg(close, 5, "slope")))

    # ── Williams Alligator (jaw=13/8, teeth=8/5, lips=5/3 SMMA) ──────────────
    def _smma(series: pd.Series, period: int) -> pd.Series:
        """Smoothed MA = EWM with alpha=1/period, same as Wilder smoothing."""
        return series.ewm(alpha=1.0 / period, adjust=False).mean()

    _add("ALLIGATOR_JAW",   _safe(lambda: _smma(close, 13)))
    _add("ALLIGATOR_TEETH", _safe(lambda: _smma(close, 8)))
    _add("ALLIGATOR_LIPS",  _safe(lambda: _smma(close, 5)))
    # Alligator direction: lips > teeth > jaw = bullish
    _add("ALLIGATOR_BULL",  _safe(lambda: (
        (df["ALLIGATOR_LIPS"] > df["ALLIGATOR_TEETH"]) &
        (df["ALLIGATOR_TEETH"] > df["ALLIGATOR_JAW"])
    ).astype(int)))
    _add("ALLIGATOR_BEAR",  _safe(lambda: (
        (df["ALLIGATOR_LIPS"] < df["ALLIGATOR_TEETH"]) &
        (df["ALLIGATOR_TEETH"] < df["ALLIGATOR_JAW"])
    ).astype(int)))

    # ── Williams Fractal ───────────────────────────────────────────────────────
    # Bearish fractal: bar[i] is the highest high of 5 consecutive bars
    # Bullish fractal: bar[i] is the lowest low of 5 consecutive bars
    _add("FRACTAL_BEAR", _safe(lambda: (
        (high > high.shift(1)) & (high > high.shift(2)) &
        (high > high.shift(-1)) & (high > high.shift(-2))
    ).astype(int)))
    _add("FRACTAL_BULL", _safe(lambda: (
        (low < low.shift(1)) & (low < low.shift(2)) &
        (low < low.shift(-1)) & (low < low.shift(-2))
    ).astype(int)))

    # ── Guppy Multiple Moving Average (12 EMAs) ───────────────────────────────
    for p in [3, 5, 8, 10, 12, 15]:
        _add(f"GMMA_S{p}", _safe(lambda p=p: close.ewm(span=p, adjust=False).mean()))
    for p in [30, 35, 40, 45, 50, 60]:
        _add(f"GMMA_L{p}", _safe(lambda p=p: close.ewm(span=p, adjust=False).mean()))
    # GMMA compression: short and long groups close together = trend change potential
    _add("GMMA_BULL", _safe(lambda: (
        (df.get("GMMA_S3", close) > df.get("GMMA_L60", close))
    ).astype(int) if "GMMA_S3" in df.columns and "GMMA_L60" in df.columns else None))

    # ── Connors RSI ───────────────────────────────────────────────────────────
    def _connors_rsi(close: pd.Series) -> pd.Series:
        """CRSI = (RSI(3) + StreakRSI(2) + ROC_percentile(100)) / 3"""
        # RSI(3)
        delta = close.diff()
        up3   = delta.clip(lower=0).rolling(3).mean()
        dn3   = (-delta).clip(lower=0).rolling(3).mean()
        rsi3  = 100 - 100 / (1 + up3 / dn3.replace(0, float("nan")))

        # Streak: count consecutive up/down days
        direction = np.sign(delta.fillna(0))
        streak_arr = np.zeros(len(direction))
        s = 0.0
        dir_arr = direction.values
        for i in range(1, len(dir_arr)):
            d = dir_arr[i]
            if d > 0:
                s = s + 1 if s > 0 else 1.0
            elif d < 0:
                s = s - 1 if s < 0 else -1.0
            else:
                s = 0.0
            streak_arr[i] = s
        streak = pd.Series(streak_arr, index=close.index)
        su = streak.clip(lower=0)
        sd = (-streak).clip(lower=0)
        su_m = su.rolling(2).mean()
        sd_m = sd.rolling(2).mean()
        streak_rsi = 100 - 100 / (1 + su_m / sd_m.replace(0, float("nan")))

        # Percentile rank of 1-bar ROC over 100 bars
        roc = close.pct_change() * 100
        pct_rank = roc.rolling(100).rank(pct=True) * 100

        return (rsi3 + streak_rsi + pct_rank) / 3

    _add("CRSI", _safe(lambda: _connors_rsi(close)))

    # ── Relative Vigor Index (pure pandas) ────────────────────────────────────
    def _rvgi(open_: pd.Series, high: pd.Series, low: pd.Series,
              close: pd.Series, length: int = 14) -> pd.Series:
        numerator   = (close - open_).rolling(4).mean()
        denominator = (high - low).rolling(4).mean().replace(0, float("nan"))
        ratio = numerator / denominator
        return ratio.rolling(length).mean()

    _add("RVGI_14", _safe(lambda: _rvgi(open_, high, low, close, 14)))

    # ── KDJ (Stochastic-based) ────────────────────────────────────────────────
    def _kdj(high: pd.Series, low: pd.Series, close: pd.Series,
             n: int = 9, signal: int = 3) -> tuple:
        lo = low.rolling(n).min()
        hi = high.rolling(n).max()
        rsv = (close - lo) / (hi - lo).replace(0, float("nan")) * 100
        k = rsv.ewm(com=signal - 1, adjust=False).mean()
        d = k.ewm(com=signal - 1, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    k, d, j = (_safe(lambda: _kdj(high, low, close)) or (None, None, None))
    _add("KDJ_K", k)
    _add("KDJ_D", d)
    _add("KDJ_J", j)

    # ── Choppiness Index ──────────────────────────────────────────────────────
    def _chop(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        tr_ = pd.concat([
            high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_sum = tr_.rolling(n).sum()
        hl_range = (high.rolling(n).max() - low.rolling(n).min()).replace(0, float("nan"))
        return 100 * np.log10(atr_sum / hl_range) / np.log10(n)

    _add("CHOP_14", _safe(lambda: _chop(high, low, close, 14)))

    # ── Vertical Horizontal Filter ────────────────────────────────────────────
    def _vhf(close: pd.Series, n: int = 28) -> pd.Series:
        hh = close.rolling(n).max()
        ll = close.rolling(n).min()
        diff = close.diff().abs().rolling(n).sum().replace(0, float("nan"))
        return (hh - ll) / diff

    _add("VHF_28", _safe(lambda: _vhf(close, 28)))

    # ── True Strength Index ───────────────────────────────────────────────────
    def _tsi(close: pd.Series, fast: int = 13, slow: int = 25) -> pd.Series:
        delta = close.diff()
        dbl_smooth = delta.ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
        dbl_abs    = delta.abs().ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
        return 100 * dbl_smooth / dbl_abs.replace(0, float("nan"))

    _add("TSI_13_25", _safe(lambda: _tsi(close, 13, 25)))

    # ── Schaff Trend Cycle ────────────────────────────────────────────────────
    def _stc(close: pd.Series, fast: int = 23, slow: int = 50, cycle: int = 10) -> pd.Series:
        macd_ = (close.ewm(span=fast, adjust=False).mean() -
                 close.ewm(span=slow, adjust=False).mean())
        def _stoch_k(series, n):
            lo = series.rolling(n).min()
            hi = series.rolling(n).max()
            return (series - lo) / (hi - lo).replace(0, float("nan")) * 100
        f1 = _stoch_k(macd_, cycle)
        pf = f1.ewm(span=3, adjust=False).mean()
        f2 = _stoch_k(pf, cycle)
        return f2.ewm(span=3, adjust=False).mean()

    _add("STC", _safe(lambda: _stc(close)))

    # ── Center of Gravity ─────────────────────────────────────────────────────
    def _cg_osc(close: pd.Series, n: int = 10) -> pd.Series:
        weights = np.arange(n, 0, -1, dtype=float)
        num = close.rolling(n).apply(lambda x: np.sum(x * weights[:len(x)]), raw=True)
        den = close.rolling(n).sum().replace(0, float("nan"))
        return -num / den

    _add("CG_10", _safe(lambda: _cg_osc(close, 10)))

    # ── Chande Forecast Oscillator ────────────────────────────────────────────
    _add("CFO_9", _safe(lambda: (close - _linreg(close, 9, "value")) / close * 100))

    # ── Ulcer Index ───────────────────────────────────────────────────────────
    def _ui(close: pd.Series, n: int = 14) -> pd.Series:
        roll_max = close.rolling(n).max()
        pct_dd = ((close - roll_max) / roll_max.replace(0, float("nan"))) * 100
        return np.sqrt((pct_dd ** 2).rolling(n).mean())

    _add("UI_14", _safe(lambda: _ui(close, 14)))

    # ── McGinley Dynamic ──────────────────────────────────────────────────────
    def _mcgd(close: pd.Series, n: int = 14) -> pd.Series:
        arr = close.values.astype(float)
        out = np.full(len(arr), np.nan)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            prev = out[i - 1]
            if np.isnan(prev):
                out[i] = arr[i]
            else:
                denom = n * (arr[i] / prev) ** 4
                out[i] = prev + (arr[i] - prev) / denom if denom != 0 else prev
        return pd.Series(out, index=close.index)

    _add("MCGD_14", _safe(lambda: _mcgd(close, 14)))

    # ── Acceleration Bands ────────────────────────────────────────────────────
    def _accbands(high: pd.Series, low: pd.Series, close: pd.Series,
                  n: int = 20) -> tuple:
        hl_avg = (high + low) / 2
        factor = 4 * (high - low) / (high + low).replace(0, float("nan"))
        upper = high * (1 + factor)
        lower = low * (1 - factor)
        mid   = close.rolling(n).mean()
        up    = upper.rolling(n).mean()
        lo    = lower.rolling(n).mean()
        return up, mid, lo

    acc_up, acc_mid, acc_lo = (_safe(lambda: _accbands(high, low, close)) or (None, None, None))
    _add("ACCB_UPPER", acc_up)
    _add("ACCB_MID",   acc_mid)
    _add("ACCB_LOWER", acc_lo)

    # ── BIAS (distance from MA as %) ──────────────────────────────────────────
    for p in [6, 14, 26]:
        ma = _safe(lambda p=p: close.ewm(span=p, adjust=False).mean())
        _add(f"BIAS_{p}", _safe(lambda p=p, ma=ma: (close / ma - 1) * 100 if ma is not None else None))

    # ── Psychological Line ────────────────────────────────────────────────────
    _add("PSL_12", _safe(lambda: (close.diff() > 0).rolling(12).mean() * 100))

    # ── AD Line ───────────────────────────────────────────────────────────────
    def _ad(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series) -> pd.Series:
        hl = (high - low).replace(0, float("nan"))
        clv = ((close - low) - (high - close)) / hl
        return (clv * vol).cumsum()

    _add("AD",    _safe(lambda: _ad(high, low, close, vol)))
    _add("ADOSC", _safe(lambda: (
        _ad(high, low, close, vol).ewm(span=3, adjust=False).mean() -
        _ad(high, low, close, vol).ewm(span=10, adjust=False).mean()
    )))

    # ── PVT (Price Volume Trend) ──────────────────────────────────────────────
    _add("PVT", _safe(lambda: (close.pct_change() * vol).cumsum()))

    # ── PVO (Percentage Volume Oscillator) ────────────────────────────────────
    vol_ema12 = vol.ewm(span=12, adjust=False).mean()
    vol_ema26 = vol.ewm(span=26, adjust=False).mean()
    _add("PVO",        _safe(lambda: (vol_ema12 - vol_ema26) / vol_ema26.replace(0, float("nan")) * 100))
    _add("PVO_signal", _safe(lambda: ((vol_ema12 - vol_ema26) / vol_ema26.replace(0, float("nan")) * 100).ewm(span=9, adjust=False).mean()))

    # ── NVI / PVI (Negative/Positive Volume Index) ────────────────────────────
    def _nvi_pvi(close: pd.Series, vol: pd.Series):
        pct = close.pct_change().fillna(0).values
        vol_chg = vol.diff().values
        n = len(close)
        nvi_arr = np.full(n, 1000.0)
        pvi_arr = np.full(n, 1000.0)
        for i in range(1, n):
            nvi_arr[i] = nvi_arr[i - 1] * (1 + pct[i]) if vol_chg[i] < 0 else nvi_arr[i - 1]
            pvi_arr[i] = pvi_arr[i - 1] * (1 + pct[i]) if vol_chg[i] > 0 else pvi_arr[i - 1]
        return pd.Series(nvi_arr, index=close.index), pd.Series(pvi_arr, index=close.index)

    nvi, pvi = _safe(lambda: _nvi_pvi(close, vol)) or (None, None)
    _add("NVI", nvi)
    _add("PVI", pvi)

    # ── BRAR (BR / AR Sentiment) ──────────────────────────────────────────────
    def _brar(open_: pd.Series, high: pd.Series, low: pd.Series,
              close: pd.Series, n: int = 14):
        prev_close = close.shift(1)
        hmo = (high - open_).clip(lower=0)
        oml = (open_ - low).clip(lower=0)
        hmc = (high - prev_close).clip(lower=0)
        lcm = (prev_close - low).clip(lower=0)
        ar = hmo.rolling(n).sum() / oml.rolling(n).sum().replace(0, float("nan")) * 100
        br = hmc.rolling(n).sum() / lcm.rolling(n).sum().replace(0, float("nan")) * 100
        return ar, br

    ar, br = _safe(lambda: _brar(open_, high, low, close, 14)) or (None, None)
    _add("AR_14", ar)
    _add("BR_14", br)

    # ── Elder's Ray Index ─────────────────────────────────────────────────────
    ema13 = close.ewm(span=13, adjust=False).mean()
    _add("BULL_POWER_13", _safe(lambda: high - ema13))
    _add("BEAR_POWER_13", _safe(lambda: low - ema13))

    # ── QQE (Quantitative Qualitative Estimation) — simplified ────────────────
    def _qqe(close: pd.Series, rsi_len: int = 14, smooth: int = 5, sf: float = 4.236) -> pd.Series:
        delta = close.diff()
        up  = delta.clip(lower=0).ewm(com=rsi_len - 1, adjust=False).mean()
        dn  = (-delta).clip(lower=0).ewm(com=rsi_len - 1, adjust=False).mean()
        rsi_ = 100 - 100 / (1 + up / dn.replace(0, float("nan")))
        rsi_smooth = rsi_.ewm(span=smooth, adjust=False).mean()
        atr_rsi = rsi_smooth.diff().abs().ewm(span=smooth, adjust=False).mean()
        fast_atr = atr_rsi * sf
        return rsi_smooth, fast_atr

    qqe_line, qqe_hist = _safe(lambda: _qqe(close)) or (None, None)
    _add("QQE_LINE", qqe_line)
    _add("QQE_HIST", qqe_hist)

    # ── SMI Ergodic ───────────────────────────────────────────────────────────
    def _smi(close: pd.Series, fast: int = 5, slow: int = 20, signal: int = 5) -> tuple:
        m = close.diff()
        dbl_m   = m.ewm(span=fast, adjust=False).mean().ewm(span=slow, adjust=False).mean()
        dbl_abs = m.abs().ewm(span=fast, adjust=False).mean().ewm(span=slow, adjust=False).mean()
        smi_  = 100 * dbl_m / dbl_abs.replace(0, float("nan"))
        sig_  = smi_.ewm(span=signal, adjust=False).mean()
        return smi_, sig_

    smi_val, smi_sig = _safe(lambda: _smi(close)) or (None, None)
    _add("SMI",        smi_val)
    _add("SMI_signal", smi_sig)

    # ── INERTIA (Linear Regression of RVI) ────────────────────────────────────
    def _inertia(close: pd.Series, n: int = 20) -> pd.Series:
        delta = close.diff()
        up  = delta.clip(lower=0).rolling(n).mean()
        dn  = (-delta).clip(lower=0).rolling(n).mean()
        rvi = up / (up + dn).replace(0, float("nan"))
        return _linreg(rvi, n, "value")

    _add("INERTIA_20", _safe(lambda: _inertia(close, 20)))

    # ── Aroon Oscillator (if not already present) ─────────────────────────────
    _add("AROONOSC_25", _safe(lambda: (
        df["AROON_up"] - df["AROON_down"]
    ) if "AROON_up" in df.columns else None))

    # ── Additional RSI variants ────────────────────────────────────────────────
    def _rsx(close: pd.Series, n: int = 14) -> pd.Series:
        """RSX — smoother version of RSI using quadratic weighted smoothing."""
        delta = close.diff()
        up = delta.clip(lower=0)
        dn = (-delta).clip(lower=0)
        f28 = up.ewm(com=n - 1, adjust=False).mean()
        f30 = dn.ewm(com=n - 1, adjust=False).mean()
        rsi_ = 100 - 100 / (1 + f28 / f30.replace(0, float("nan")))
        # Second EMA smoothing for RSX
        return rsi_.ewm(span=n // 2 or 1, adjust=False).mean()

    _add("RSX_14", _safe(lambda: _rsx(close, 14)))

    # ── TTM Trend ─────────────────────────────────────────────────────────────
    def _ttm_trend(close: pd.Series, n: int = 6) -> pd.Series:
        avg = (close + close.shift(1) + close.shift(2)) / 3
        trend = (close > avg.rolling(n).mean()).map({True: 1, False: -1})
        return trend

    _add("TTM_TREND_6", _safe(lambda: _ttm_trend(close, 6)))

    # ── PMAX (Price Momentum And Oscillator Max) ───────────────────────────────
    def _pmax(high: pd.Series, low: pd.Series, close: pd.Series,
              vol: pd.Series, period: int = 10, mult: float = 3.0) -> pd.Series:
        """Simplified PMAX = SuperTrend direction with VWMA instead of close."""
        vwma = (close * vol).rolling(period).sum() / vol.rolling(period).sum()
        _, direction = _supertrend(high, low, vwma.fillna(close), period, mult)
        return direction

    _add("PMAX_dir", _safe(lambda: _pmax(high, low, close, vol)))

    # ── Keltner + BB Squeeze Momentum ─────────────────────────────────────────
    # (Adds squeeze momentum histogram if KC and BB are available)
    if "BB_upper_20" in df.columns and "KC_upper" in df.columns:
        _add("SQUEEZE_HIST", _safe(lambda: (
            close - (df["KC_upper"] + df["KC_lower"]) / 2
        )))

    # ── Volume Oscillator ─────────────────────────────────────────────────────
    _add("VOL_OSC", _safe(lambda: (
        vol.ewm(span=5, adjust=False).mean() - vol.ewm(span=10, adjust=False).mean()
    ) / vol.ewm(span=10, adjust=False).mean().replace(0, float("nan")) * 100))

    # ── Coppock Curve (pure pandas) ───────────────────────────────────────────
    if "COPPOCK" not in df.columns:
        _add("COPPOCK", _safe(lambda: (
            (close.pct_change(14) + close.pct_change(11)) * 100
        ).rolling(10).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
        )))

    # ── Triangular MA ─────────────────────────────────────────────────────────
    def _trima(close: pd.Series, n: int) -> pd.Series:
        half = (n + 1) // 2
        return close.rolling(half).mean().rolling(half).mean()

    _add("TRIMA_20", _safe(lambda: _trima(close, 20)))
    _add("TRIMA_50", _safe(lambda: _trima(close, 50)))

    # ── Zero-Lag EMA ──────────────────────────────────────────────────────────
    def _zlema(close: pd.Series, n: int) -> pd.Series:
        lag = (n - 1) // 2
        adjusted = 2 * close - close.shift(lag)
        return adjusted.ewm(span=n, adjust=False).mean()

    _add("ZLEMA_20", _safe(lambda: _zlema(close, 20)))
    _add("ZLEMA_50", _safe(lambda: _zlema(close, 50)))

    # ── Arnaud Legoux MA ──────────────────────────────────────────────────────
    def _alma(close: pd.Series, n: int = 9, sigma: float = 6.0, offset: float = 0.85) -> pd.Series:
        m = offset * (n - 1)
        s = n / sigma
        weights = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
        weights /= weights.sum()
        return close.rolling(n).apply(lambda x: np.dot(x, weights[::-1]), raw=True)

    _add("ALMA_9",  _safe(lambda: _alma(close, 9)))
    _add("ALMA_21", _safe(lambda: _alma(close, 21)))

    # ── T3 (Triple Exponential MA) ────────────────────────────────────────────
    def _t3(close: pd.Series, n: int = 5, vf: float = 0.7) -> pd.Series:
        c1 = -(vf ** 3)
        c2 = 3 * vf ** 2 + 3 * vf ** 3
        c3 = -6 * vf ** 2 - 3 * vf - 3 * vf ** 3
        c4 = 1 + 3 * vf + vf ** 3 + 3 * vf ** 2
        e1 = close.ewm(span=n, adjust=False).mean()
        e2 = e1.ewm(span=n, adjust=False).mean()
        e3 = e2.ewm(span=n, adjust=False).mean()
        e4 = e3.ewm(span=n, adjust=False).mean()
        e5 = e4.ewm(span=n, adjust=False).mean()
        e6 = e5.ewm(span=n, adjust=False).mean()
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    _add("T3_5",  _safe(lambda: _t3(close, 5)))
    _add("T3_10", _safe(lambda: _t3(close, 10)))

    # ── VIDYA (Variable Index Dynamic Average) ────────────────────────────────
    def _vidya(close: pd.Series, n: int = 14, smooth: int = 12) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        dn = (-delta).clip(lower=0).rolling(n).mean()
        cmo = (up - dn) / (up + dn).replace(0, float("nan"))
        alpha = 2 / (smooth + 1)
        arr = close.values.astype(float)
        cmo_arr = cmo.values
        out = np.full(len(arr), np.nan)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            if np.isnan(out[i - 1]) or np.isnan(cmo_arr[i]):
                out[i] = arr[i]
            else:
                k = alpha * abs(cmo_arr[i])
                out[i] = k * arr[i] + (1 - k) * out[i - 1]
        return pd.Series(out, index=close.index)

    _add("VIDYA_14", _safe(lambda: _vidya(close, 14)))

    # ── Fibonacci Weighted MA ─────────────────────────────────────────────────
    def _fwma(close: pd.Series, n: int = 10) -> pd.Series:
        def _fib_weights(k):
            a, b = 1, 1
            fibs = [a]
            for _ in range(k - 1):
                a, b = b, a + b
                fibs.append(a)
            w = np.array(fibs, dtype=float)
            return w / w.sum()
        w = _fib_weights(n)
        return close.rolling(n).apply(lambda x: np.dot(x, w[::-1]), raw=True)

    _add("FWMA_10", _safe(lambda: _fwma(close, 10)))

    # ── Klinger Volume Oscillator ─────────────────────────────────────────────
    def _kvo(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series,
             fast: int = 34, slow: int = 55) -> pd.Series:
        typ = (high + low + close) / 3
        trend = np.sign(typ.diff()).values
        dm = (high - low).values
        n = len(close)
        cm_arr = np.zeros(n)
        for i in range(1, n):
            cm_arr[i] = cm_arr[i - 1] + dm[i] if trend[i] == trend[i - 1] else dm[i - 1] + dm[i]
        cm = pd.Series(cm_arr, index=close.index).replace(0, float("nan"))
        sv = pd.Series(trend, index=close.index) * vol * abs(2 * (high - low) / cm - 1)
        return sv.ewm(span=fast, adjust=False).mean() - sv.ewm(span=slow, adjust=False).mean()

    _add("KVO", _safe(lambda: _kvo(high, low, close, vol)))

    # ── QSTICK (body direction smoothed) ─────────────────────────────────────
    _add("QSTICK_8", _safe(lambda: (close - open_).rolling(8).mean()))

    # ── Historical Volatility variants ────────────────────────────────────────
    log_ret = _safe(lambda: np.log(close / close.shift(1)))
    if log_ret is not None:
        _add("HV_10", _safe(lambda: log_ret.rolling(10).std() * np.sqrt(365 * 24)))
        _add("HV_30", _safe(lambda: log_ret.rolling(30).std() * np.sqrt(365 * 24)))
        _add("HV_60", _safe(lambda: log_ret.rolling(60).std() * np.sqrt(365 * 24)))

    # ── Volatility Ratio (current ATR vs avg ATR) ─────────────────────────────
    if "ATR_14" in df.columns:
        _add("VOL_RATIO_14", _safe(lambda: df["ATR_14"] / df["ATR_14"].rolling(100).mean()))

    # ── RSI divergence proxy (price makes new high, RSI doesn't) ─────────────
    if "RSI_14" in df.columns:
        _add("RSI_DIVERG", _safe(lambda: (
            (close > close.rolling(14).max().shift(1)) &
            (df["RSI_14"] < df["RSI_14"].rolling(14).max().shift(1))
        ).astype(int)))

    # ── MACD cross signal ─────────────────────────────────────────────────────
    if "MACD" in df.columns and "MACD_signal" in df.columns:
        _add("MACD_cross", _safe(lambda: (
            (df["MACD"] > df["MACD_signal"]).astype(int) -
            (df["MACD"].shift(1) > df["MACD_signal"].shift(1)).astype(int)
        )))

    # ── NATR for multiple periods ──────────────────────────────────────────────
    for p in [7, 21]:
        if f"ATR_{p}" in df.columns:
            _add(f"NATR_{p}", _safe(lambda p=p: (df[f"ATR_{p}"] / close) * 100))

    # ── Channel Width (Donchian / Keltner / BB) ───────────────────────────────
    if "DC_upper" in df.columns and "DC_lower" in df.columns:
        _add("DC_WIDTH", _safe(lambda: (df["DC_upper"] - df["DC_lower"]) / close * 100))
    if "KC_upper" in df.columns and "KC_lower" in df.columns:
        _add("KC_WIDTH", _safe(lambda: (df["KC_upper"] - df["KC_lower"]) / close * 100))

    # ── Trend Strength Index (close direction consistency) ─────────────────────
    _add("TREND_STR_14", _safe(lambda: (
        (close.diff() > 0).rolling(14).sum() / 14.0 * 100
    )))

    # ── Price vs VWAP (distance %) ────────────────────────────────────────────
    if "VWAP" in df.columns:
        _add("CLOSE_VS_VWAP", _safe(lambda: (close / df["VWAP"] - 1) * 100))

    # ── OBV normalized (z-score of OBV) ───────────────────────────────────────
    if "OBV" in df.columns:
        _add("OBV_ZSCORE", _safe(lambda: (
            (df["OBV"] - df["OBV"].rolling(20).mean()) /
            df["OBV"].rolling(20).std().replace(0, float("nan"))
        )))

    # ── Holt-Winters style double exponential ────────────────────────────────
    def _hwma(close: pd.Series, na: float = 0.2, nb: float = 0.1, nc: float = 0.1) -> pd.Series:
        arr = close.values.astype(float)
        out = np.full(len(arr), np.nan)
        if len(arr) < 2:
            return pd.Series(out, index=close.index)
        out[0] = arr[0]
        b = arr[1] - arr[0]
        for i in range(1, len(arr)):
            prev = out[i - 1]
            f = na * arr[i] + (1 - na) * (prev + b)
            b = nb * (f - prev) + (1 - nb) * b
            out[i] = nc * arr[i] + (1 - nc) * f
        return pd.Series(out, index=close.index)

    _add("HWMA", _safe(lambda: _hwma(close)))

    # ── DEMA / TEMA extended periods ──────────────────────────────────────────
    for p in [9, 21]:
        e1 = close.ewm(span=p, adjust=False).mean()
        e2 = e1.ewm(span=p, adjust=False).mean()
        e3 = e2.ewm(span=p, adjust=False).mean()
        _add(f"DEMA_ext_{p}", _safe(lambda e1=e1, e2=e2: 2 * e1 - e2))
        _add(f"TEMA_ext_{p}", _safe(lambda e1=e1, e2=e2, e3=e3: 3 * e1 - 3 * e2 + e3))

    # ── Pascal Weighted MA ────────────────────────────────────────────────────
    def _pwma(close: pd.Series, n: int = 10) -> pd.Series:
        from math import comb
        weights = np.array([comb(n - 1, i) for i in range(n)], dtype=float)
        weights /= weights.sum()
        return close.rolling(n).apply(lambda x: np.dot(x, weights[::-1]), raw=True)

    _add("PWMA_10", _safe(lambda: _pwma(close, 10)))

    # ── Symmetric Weighted MA ────────────────────────────────────────────────
    def _swma(close: pd.Series) -> pd.Series:
        weights = np.array([1, 2, 2, 1], dtype=float)
        weights /= weights.sum()
        return close.rolling(4).apply(lambda x: np.dot(x, weights[::-1]), raw=True)

    _add("SWMA", _safe(lambda: _swma(close)))

    # ── Hilo Activator ────────────────────────────────────────────────────────
    _add("HILO_HIGH", _safe(lambda: high.rolling(13).mean()))
    _add("HILO_LOW",  _safe(lambda: low.rolling(21).mean()))
    _add("HILO_dir",  _safe(lambda: (
        (close > df["HILO_HIGH"]).astype(int) -
        (close < df["HILO_LOW"]).astype(int)
    ) if "HILO_HIGH" in df.columns and "HILO_LOW" in df.columns else None))

    # ── Decay ─────────────────────────────────────────────────────────────────
    _add("LIN_DECAY_5", _safe(lambda: close.rolling(5).apply(
        lambda x: x[-1] * (1 - np.arange(1, 6)[::-1] / 5).mean(), raw=True
    )))

    # ── Absolute Moving Average Trend (AMAT) ──────────────────────────────────
    _add("AMAT_fast", _safe(lambda: (close.ewm(span=8, adjust=False).mean() >
                                     close.ewm(span=21, adjust=False).mean()).astype(int)))

    # ── Price Rate of Change variants ─────────────────────────────────────────
    for p in [5, 14, 30]:
        _add(f"ROC_{p}", _safe(lambda p=p: (close / close.shift(p) - 1) * 100))

    # ── Stochastic RSI extra smoothing ────────────────────────────────────────
    if "STOCHRSI_K" in df.columns:
        _add("STOCHRSI_K_smooth", _safe(lambda: df["STOCHRSI_K"].rolling(3).mean()))

    # ── Supertrend extra configs ──────────────────────────────────────────────
    for period, mult in [(7, 3.0), (10, 2.0), (14, 3.0), (20, 2.0)]:
        col_dir = f"SUPERT_dir_{period}_{int(mult)}"
        if col_dir not in df.columns:
            st, st_dir = _safe(lambda p=period, m=mult: _supertrend(high, low, close, p, m)) or (None, None)
            if st_dir is not None:
                _add(f"SUPERT_{period}_{int(mult)}", st)
                _add(col_dir, st_dir)

    # ── Average Directional Index extra periods ────────────────────────────────
    try:
        import ta.trend as _tr
        for p in [7, 21]:
            adx_obj = _tr.ADXIndicator(high, low, close, window=p, fillna=False)
            _add(f"ADX_{p}", adx_obj.adx())
            _add(f"DMP_{p}", adx_obj.adx_pos())
            _add(f"DMN_{p}", adx_obj.adx_neg())
    except Exception:
        pass


# ── Multi-Timeframe Feature Injection ────────────────────────────────────────

# Key HTF columns to inject (forward-filled onto primary TF candles)
HTF_INJECT_COLS = [
    "RSI_14", "RSI_7",
    "EMA_20", "EMA_50", "EMA_200",
    "MACD_hist", "MACD",
    "ADX_14",
    "BB_pct_20", "BB_width_20",
    "ATR_14", "NATR_14",
    "STOCH_K",
    "SUPERT_dir",
    "CCI_20",
    "MFI_14",
    "volume_ratio",
    "EMA50_slope",
    "HIGH_VOL_REGIME",
    "PRICE_RANGE_PCT",
    "ema_20_50_cross",
    "ema_50_200_cross",
]


def inject_htf_features(primary_df: pd.DataFrame, htf_df: pd.DataFrame,
                         htf_label: str,
                         base_strategy: "BaseStrategy | None" = None) -> pd.DataFrame:
    """
    Inject Higher Timeframe (HTF) indicators onto the primary TF DataFrame.

    Each HTF indicator column is prefixed HTF_{htf_label}_ and forward-filled
    so every primary TF candle knows its HTF context at that moment.

    Also adds derived contextual columns:
      HTF_{label}_trend_dir  — +1 if close > EMA_50 on HTF, -1 otherwise
      HTF_{label}_regime     — 1=trending_up, -1=trending_down, 0=ranging

    Args:
        primary_df: Primary timeframe OHLCV + indicators DataFrame.
        htf_df: Higher timeframe OHLCV DataFrame (must have 'timestamp' column).
        htf_label: e.g. '1h', '4h', '1d'.
        base_strategy: If provided and htf_df lacks indicators, they are computed.

    Returns:
        primary_df with HTF_ prefixed columns added.
    """
    if htf_df is None or len(htf_df) < 20:
        return primary_df

    htf = htf_df.copy()

    # Compute indicators on HTF if not present
    if "RSI_14" not in htf.columns and base_strategy is not None:
        htf = base_strategy.populate_indicators(htf)

    # Derived trend context
    if "EMA_50" in htf.columns:
        htf[f"_trend_dir"] = (htf["close"] >= htf["EMA_50"]).map({True: 1, False: -1})
    if "EMA50_slope" in htf.columns:
        slope = htf["EMA50_slope"]
        htf["_regime"] = 0
        htf.loc[slope > 0.05, "_regime"]  =  1
        htf.loc[slope < -0.05, "_regime"] = -1

    # Build the set of columns to inject
    cols_to_inject = []
    for c in HTF_INJECT_COLS:
        if c in htf.columns:
            cols_to_inject.append(c)
    for derived in ["_trend_dir", "_regime"]:
        if derived in htf.columns:
            cols_to_inject.append(derived)

    if not cols_to_inject:
        return primary_df

    # Rename to HTF_ prefix
    htf_sub = htf[["timestamp"] + cols_to_inject].copy()
    rename_map = {c: f"HTF_{htf_label}_{c.lstrip('_')}" for c in cols_to_inject}
    htf_sub = htf_sub.rename(columns=rename_map)

    # Sort both by timestamp
    primary_sorted = primary_df.sort_values("timestamp").reset_index(drop=True)
    htf_sub = htf_sub.sort_values("timestamp").reset_index(drop=True)

    # Merge-asof: for each primary bar, use the latest HTF bar that has completed
    merged = pd.merge_asof(primary_sorted, htf_sub, on="timestamp", direction="backward")

    # Forward-fill any remaining NaN (early bars before first HTF bar)
    htf_new_cols = list(rename_map.values())
    merged[htf_new_cols] = merged[htf_new_cols].ffill()

    return merged


def get_htf_col(htf_label: str, base_col: str) -> str:
    """Return the HTF column name for a given base column and timeframe label."""
    return f"HTF_{htf_label}_{base_col}"


# ── Entry signal handling ─────────────────────────────────────────────────────

class CodeStrategy(BaseStrategy):
    """
    Strategy built from user-supplied Python code.
    The code string must define populate_entry_signal(df) and optionally
    populate_exit_signal(df), and may override populate_indicators(df).
    """

    name = "CustomCodeStrategy"

    def __init__(self, code: str, params: dict | None = None):
        super().__init__(params)
        self._code = code
        self._ns: dict = {}
        exec(compile(code, "<strategy>", "exec"), self._ns)  # noqa: S102

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        fn = self._ns.get("populate_entry_signal")
        if fn:
            return fn(df)
        return super().populate_entry_signal(df)

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        fn = self._ns.get("populate_exit_signal")
        if fn:
            return fn(df)
        return super().populate_exit_signal(df)

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fn = self._ns.get("populate_indicators")
        if fn:
            return fn(super().populate_indicators(df))
        return super().populate_indicators(df)


DEFAULT_STRATEGY_CODE = '''
# CryptoAlgoFinder – Custom Strategy Template
# ───────────────────────────────────────────────
# The DataFrame 'df' already contains all indicators added by BaseStrategy.
# Available columns include: open, high, low, close, volume, timestamp,
#   EMA_8/20/50/200, HMA_20/50, TEMA_21, DEMA_21,
#   RSI_7/14/21, MACD/MACD_signal/MACD_hist, CMO_14, FISHER,
#   BB_upper_20/BB_lower_20/BB_pct_20, ATR_14, NATR_14, ADX_14,
#   CCI_20, SUPERT_dir, PSAR_dir, BB_SQUEEZE,
#   STOCH_K/D, MFI_14, WILLR_14, volume_ratio,
#   PRICE_RANGE_PCT, HV_20, EMA50_slope, HIGH_VOL_REGIME,
#   CDL_HAMMER, CDL_MORNING_STAR, CDL_3_WHITE_SOLDIERS, and 25+ more.
#
# Required: populate_entry_signal(df) → return df with column \'entry_signal\' (1/0)
# Optional: populate_exit_signal(df)  → return df with column \'exit_signal\' (1/0)

def populate_entry_signal(df):
    """
    Buy when:
    - RSI 14 is oversold (< 35)
    - Price is above EMA 50 (uptrend filter)
    - MACD histogram is positive (momentum)
    - ADX > 20 (trending market)
    """
    df["entry_signal"] = (
        (df["RSI_14"] < 35) &
        (df["close"] > df["EMA_50"]) &
        (df["MACD_hist"] > 0) &
        (df["ADX_14"] > 20)
    ).astype(int)
    return df


def populate_exit_signal(df):
    """
    Sell when:
    - RSI 14 is overbought (> 70)
    - OR EMA 20 crosses below EMA 50
    """
    df["exit_signal"] = (
        (df["RSI_14"] > 70) |
        (df["ema_20_50_cross"] == -1)
    ).astype(int)
    return df
'''
