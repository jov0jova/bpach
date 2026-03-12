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
            df["ICH_above_cloud"] = ((close > df["ICH_senkou_a"]) & (close > df["ICH_senkou_b"])).astype(int)
            df["ICH_below_cloud"] = ((close < df["ICH_senkou_a"]) & (close < df["ICH_senkou_b"])).astype(int)

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
            df["PSAR_dir"] = ((close > psar_val).astype(int) * 2 - 1)

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
    """
    hl2 = (high + low) / 2
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    direction  = pd.Series(0, index=close.index, dtype=int)

    for i in range(1, len(close)):
        idx     = close.index[i]
        idx_p   = close.index[i - 1]

        lb = lower_band.iloc[i]
        ub = upper_band.iloc[i]

        # Final lower / upper bands with carry-forward logic
        final_lb = lb if lb > supertrend.get(idx_p, lb) or close.iloc[i - 1] < supertrend.get(idx_p, lb) else supertrend.get(idx_p, lb)
        final_ub = ub if ub < supertrend.get(idx_p, ub) or close.iloc[i - 1] > supertrend.get(idx_p, ub) else supertrend.get(idx_p, ub)

        prev_st  = supertrend.get(idx_p, final_ub)
        prev_dir = direction.iloc[i - 1]

        if prev_st == final_ub:
            if close.iloc[i] > final_ub:
                supertrend[idx] = final_lb
                direction[idx]  = 1
            else:
                supertrend[idx] = final_ub
                direction[idx]  = -1
        else:
            if close.iloc[i] < final_lb:
                supertrend[idx] = final_ub
                direction[idx]  = -1
            else:
                supertrend[idx] = final_lb
                direction[idx]  = 1

    return supertrend, direction


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
    return fisher.fillna(method="ffill")


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

    # Days since new high / low (normalized)
    bars_since_high = pd.Series(index=close.index, dtype=float)
    bars_since_low  = pd.Series(index=close.index, dtype=float)
    peak = close.iloc[0]
    trough = close.iloc[0]
    since_high = 0
    since_low = 0
    for i, (idx, c_val, h_val, l_val) in enumerate(zip(close.index, close.values, high.values, low.values)):
        if h_val >= peak:
            peak = h_val
            since_high = 0
        else:
            since_high += 1
        if l_val <= trough:
            trough = l_val
            since_low = 0
        else:
            since_low += 1
        bars_since_high.iloc[i] = since_high
        bars_since_low.iloc[i]  = since_low

    df["BARS_SINCE_HIGH"] = bars_since_high
    df["BARS_SINCE_LOW"]  = bars_since_low

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
    merged[htf_new_cols] = merged[htf_new_cols].fillna(method="ffill")

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
