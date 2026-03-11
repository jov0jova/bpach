"""
BaseStrategy: adds every major technical analysis indicator to a DataFrame.
All custom strategies inherit from this and override populate_entry_signal()
and populate_exit_signal().

Uses the `ta` library (https://github.com/bukosabino/ta) which is pure Python
and installs cleanly on all platforms including Windows + Python 3.11.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class BaseStrategy:
    """
    Base class for all CryptoAlgoFinder strategies.
    Provides populate_indicators() which adds 80+ indicators via the `ta` library.
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
        Add the full suite of technical indicators using the `ta` library.
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

        try:
            # ── Trend ─────────────────────────────────────────────────────────
            for period in [8, 13, 20, 21, 50, 100, 200]:
                df[f"EMA_{period}"] = tr.EMAIndicator(close, window=period, fillna=False).ema_indicator()
                df[f"SMA_{period}"] = tr.SMAIndicator(close, window=period, fillna=False).sma_indicator()

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

            # Ichimoku
            ichi = tr.IchimokuIndicator(high, low, window1=9, window2=26, window3=52, fillna=False)
            df["ICH_tenkan"]   = ichi.ichimoku_conversion_line()
            df["ICH_kijun"]    = ichi.ichimoku_base_line()
            df["ICH_senkou_a"] = ichi.ichimoku_a()
            df["ICH_senkou_b"] = ichi.ichimoku_b()

            # Aroon
            aroon = tr.AroonIndicator(high, low, window=25, fillna=False)
            df["AROON_up"]   = aroon.aroon_up()
            df["AROON_down"] = aroon.aroon_down()

            # PSAR (Parabolic SAR)
            psar_obj = tr.PSARIndicator(high, low, close, step=0.02, max_step=0.2, fillna=False)
            df["PSAR_up"]   = psar_obj.psar_up()
            df["PSAR_down"] = psar_obj.psar_down()

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

            # Awesome Oscillator
            df["AO"] = mom.AwesomeOscillatorIndicator(high, low, window1=5, window2=34, fillna=False).awesome_oscillator()

            # KAMA
            df["KAMA"] = mom.KAMAIndicator(close, window=10, pow1=2, pow2=30, fillna=False).kama()

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

            dc = vol_mod.DonchianChannel(high, low, close, window=20, fillna=False)
            df["DC_upper"]  = dc.donchian_channel_hband()
            df["DC_lower"]  = dc.donchian_channel_lband()
            df["DC_middle"] = dc.donchian_channel_mband()

            # ── Volume ────────────────────────────────────────────────────────
            df["OBV"]    = volm.OnBalanceVolumeIndicator(close, vol, fillna=False).on_balance_volume()
            df["MFI_14"] = volm.MFIIndicator(high, low, close, vol, window=14, fillna=False).money_flow_index()
            df["CMF_20"] = volm.ChaikinMoneyFlowIndicator(high, low, close, vol, window=20, fillna=False).chaikin_money_flow()
            df["VWAP"]   = volm.VolumeWeightedAveragePrice(high, low, close, vol, window=14, fillna=False).volume_weighted_average_price()

            # ── Derived / Price-Action ────────────────────────────────────────
            df["candle_body"]      = (close - df["open"]).abs()
            df["candle_range"]     = high - low
            df["body_pct"]         = df["candle_body"] / df["candle_range"].replace(0, float("nan"))
            df["upper_wick"]       = high - df[["open", "close"]].max(axis=1)
            df["lower_wick"]       = df[["open", "close"]].min(axis=1) - low
            df["close_pct_change"] = close.pct_change() * 100

            vol_ma = vol.rolling(20).mean()
            df["volume_ratio"] = vol / vol_ma.replace(0, float("nan"))

            # EMA cross signals: +1 = bullish cross, -1 = bearish cross, 0 = no cross
            df["ema_20_50_cross"] = (
                (df["EMA_20"] > df["EMA_50"]).astype(int) -
                (df["EMA_20"].shift(1) > df["EMA_50"].shift(1)).astype(int)
            )
            df["ema_50_200_cross"] = (
                (df["EMA_50"] > df["EMA_200"]).astype(int) -
                (df["EMA_50"].shift(1) > df["EMA_200"].shift(1)).astype(int)
            )

            # Candlestick patterns (pure price-action, no TA-Lib required)
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


def _add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure price-action candlestick pattern detection (no TA-Lib required).
    Each column is +100 (bullish), -100 (bearish), or 0.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body  = (c - o).abs()
    rng   = (h - l).replace(0, float("nan"))
    upper = h - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - l

    # Doji: body < 10% of range
    df["CDL_DOJI"] = ((body / rng) < 0.1).map({True: 100, False: 0})

    # Hammer: lower wick > 2× body, upper wick < body, bullish
    df["CDL_HAMMER"] = (
        (lower > 2 * body) & (upper < body) & (c > o)
    ).map({True: 100, False: 0})

    # Shooting Star: upper wick > 2× body, lower wick < body, bearish
    df["CDL_SHOOTING_STAR"] = (
        (upper > 2 * body) & (lower < body) & (c < o)
    ).map({True: -100, False: 0})

    # Bullish Engulfing
    prev_o, prev_c = o.shift(1), c.shift(1)
    df["CDL_BULL_ENGULFING"] = (
        (prev_c < prev_o) & (c > o) & (o < prev_c) & (c > prev_o)
    ).map({True: 100, False: 0})

    # Bearish Engulfing
    df["CDL_BEAR_ENGULFING"] = (
        (prev_c > prev_o) & (c < o) & (o > prev_c) & (c < prev_o)
    ).map({True: -100, False: 0})

    # Bullish Harami
    prev_body = (prev_c - prev_o).abs()
    df["CDL_BULL_HARAMI"] = (
        (prev_c < prev_o) & (c > o) &
        (o > prev_c) & (c < prev_o) &
        (body < prev_body * 0.5)
    ).map({True: 100, False: 0})

    # Dragonfly Doji: very small body + upper wick
    df["CDL_DRAGONFLY"] = (
        ((body / rng) < 0.1) & (upper < rng * 0.1)
    ).map({True: 100, False: 0})

    # Gravestone Doji: very small body + lower wick
    df["CDL_GRAVESTONE"] = (
        ((body / rng) < 0.1) & (lower < rng * 0.1)
    ).map({True: -100, False: 0})

    return df


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
#   EMA_8/20/50/200, RSI_7/14/21, MACD/MACD_signal/MACD_hist,
#   BB_upper_20/BB_lower_20/BB_pct_20, ATR_14, ADX_14, STOCH_K/D,
#   OBV, VWAP, MFI_14, WILLR_14, volume_ratio, CDL_HAMMER, and more.
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
