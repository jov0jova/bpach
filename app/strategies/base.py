"""
BaseStrategy: adds every major technical analysis indicator to a DataFrame.
All custom strategies inherit from this and override populate_entry_signal()
and populate_exit_signal().
"""
import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


class BaseStrategy:
    """
    Base class for all CryptoAlgoFinder strategies.
    Provides populate_indicators() which adds ~100 indicators via pandas-ta.
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
        Add the full suite of technical indicators.
        Column naming follows pandas-ta convention.
        """
        if len(df) < 50:
            return df

        try:
            # ── Trend ─────────────────────────────────────────────────────────
            for period in [8, 13, 20, 21, 50, 100, 200]:
                df[f"EMA_{period}"] = ta.ema(df["close"], length=period)
                df[f"SMA_{period}"] = ta.sma(df["close"], length=period)

            # Supertrend
            st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
            if st is not None:
                df["SUPERT"] = st.get("SUPERT_10_3.0")
                df["SUPERT_dir"] = st.get("SUPERTd_10_3.0")

            # PSAR
            psar = ta.psar(df["high"], df["low"], df["close"])
            if psar is not None and not psar.empty:
                df["PSAR_long"] = psar.get("PSARl_0.02_0.2")
                df["PSAR_short"] = psar.get("PSARs_0.02_0.2")

            # Ichimoku (simplified)
            ich = ta.ichimoku(df["high"], df["low"], df["close"])
            if ich is not None and len(ich) >= 2:
                base = ich[0]
                if base is not None and not base.empty:
                    df["ICH_tenkan"] = base.get("ITS_9")
                    df["ICH_kijun"] = base.get("IKS_26")
                    df["ICH_senkou_a"] = base.get("ISA_9")
                    df["ICH_senkou_b"] = base.get("ISB_26")

            # ── Momentum ──────────────────────────────────────────────────────
            for period in [7, 14, 21]:
                df[f"RSI_{period}"] = ta.rsi(df["close"], length=period)

            # Stochastic RSI
            stoch_rsi = ta.stochrsi(df["close"], length=14)
            if stoch_rsi is not None and not stoch_rsi.empty:
                df["STOCHRSI_K"] = stoch_rsi.get("STOCHRSIk_14_14_3_3")
                df["STOCHRSI_D"] = stoch_rsi.get("STOCHRSId_14_14_3_3")

            # Stochastic
            stoch = ta.stoch(df["high"], df["low"], df["close"])
            if stoch is not None and not stoch.empty:
                df["STOCH_K"] = stoch.get("STOCHk_14_3_3")
                df["STOCH_D"] = stoch.get("STOCHd_14_3_3")

            # MACD
            macd = ta.macd(df["close"])
            if macd is not None and not macd.empty:
                df["MACD"] = macd.get("MACD_12_26_9")
                df["MACD_signal"] = macd.get("MACDs_12_26_9")
                df["MACD_hist"] = macd.get("MACDh_12_26_9")

            # MFI
            df["MFI_14"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

            # CCI
            df["CCI_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)

            # Williams %R
            df["WILLR_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)

            # ROC
            df["ROC_10"] = ta.roc(df["close"], length=10)

            # ── Volatility ────────────────────────────────────────────────────
            # Bollinger Bands
            for period in [14, 20]:
                bb = ta.bbands(df["close"], length=period)
                if bb is not None and not bb.empty:
                    df[f"BB_upper_{period}"] = bb.get(f"BBU_{period}_2.0")
                    df[f"BB_mid_{period}"] = bb.get(f"BBM_{period}_2.0")
                    df[f"BB_lower_{period}"] = bb.get(f"BBL_{period}_2.0")
                    df[f"BB_width_{period}"] = bb.get(f"BBB_{period}_2.0")
                    df[f"BB_pct_{period}"] = bb.get(f"BBP_{period}_2.0")

            # ATR
            for period in [7, 14, 21]:
                df[f"ATR_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)

            # Keltner Channels
            kc = ta.kc(df["high"], df["low"], df["close"])
            if kc is not None and not kc.empty:
                df["KC_upper"] = kc.get("KCUe_20_2")
                df["KC_lower"] = kc.get("KCLe_20_2")

            # Donchian Channels
            dc = ta.donchian(df["high"], df["low"])
            if dc is not None and not dc.empty:
                df["DC_upper"] = dc.get("DCU_20_20")
                df["DC_lower"] = dc.get("DCL_20_20")

            # ── Volume ────────────────────────────────────────────────────────
            df["OBV"] = ta.obv(df["close"], df["volume"])
            df["VWMA_20"] = ta.vwma(df["close"], df["volume"], length=20)

            ad = ta.ad(df["high"], df["low"], df["close"], df["volume"])
            if ad is not None:
                df["AD"] = ad

            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx is not None and not adx.empty:
                df["ADX_14"] = adx.get("ADX_14")
                df["DMP_14"] = adx.get("DMP_14")
                df["DMN_14"] = adx.get("DMN_14")

            # VWAP (intra-day approximation)
            df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

            # ── Candlestick patterns ──────────────────────────────────────────
            # pandas-ta CDL patterns return +100, 0, or -100
            pattern_funcs = {
                "CDL_DOJI": ta.cdl_doji,
                "CDL_HAMMER": ta.cdl_hammer,
                "CDL_SHOOTING_STAR": ta.cdl_shootingstar,
                "CDL_ENGULFING": ta.cdl_inside,   # bullish/bearish engulf via inside
                "CDL_MORNING_STAR": ta.cdl_morningstar,
                "CDL_EVENING_STAR": ta.cdl_eveningstar,
                "CDL_HARAMI": ta.cdl_harami,
                "CDL_DRAGONFLY": ta.cdl_dragonfly_doji,
                "CDL_GRAVESTONE": ta.cdl_gravestone_doji,
                "CDL_THREE_BLACK": ta.cdl_3blackcrows,
                "CDL_THREE_WHITE": ta.cdl_3whitesoldiers,
            }
            for col_name, fn in pattern_funcs.items():
                try:
                    result = fn(df["open"], df["high"], df["low"], df["close"])
                    if result is not None:
                        if isinstance(result, pd.DataFrame):
                            df[col_name] = result.iloc[:, 0]
                        else:
                            df[col_name] = result
                except Exception:
                    pass  # Some patterns may not be available in all versions

            # ── Derived / Price-Action ────────────────────────────────────────
            df["candle_body"] = abs(df["close"] - df["open"])
            df["candle_range"] = df["high"] - df["low"]
            df["body_pct"] = df["candle_body"] / df["candle_range"].replace(0, float("nan"))
            df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
            df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
            df["close_pct_change"] = df["close"].pct_change() * 100

            # Volume ratio vs 20-bar average
            vol_ma = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / vol_ma.replace(0, float("nan"))

            # EMA cross signals
            df["ema_20_50_cross"] = (
                (df["EMA_20"] > df["EMA_50"]).astype(int) -
                (df["EMA_20"].shift(1) > df["EMA_50"].shift(1)).astype(int)
            )  # +1 = bullish cross, -1 = bearish cross

            df["ema_50_200_cross"] = (
                (df["EMA_50"] > df["EMA_200"]).astype(int) -
                (df["EMA_50"].shift(1) > df["EMA_200"].shift(1)).astype(int)
            )

        except Exception as e:
            logger.warning("populate_indicators error: %s", e)

        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Override in subclasses. Must add column 'entry_signal' (1=buy, 0=no).
        Default: EMA20 > EMA50 AND RSI14 < 70 AND close > VWAP.
        """
        if "EMA_20" not in df.columns or "EMA_50" not in df.columns:
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
#   BB_upper_20/BB_lower_20, ATR_14, ADX_14, STOCH_K/D, OBV, VWAP, and more.
#
# Required: populate_entry_signal(df) → return df with column 'entry_signal' (1/0)
# Optional: populate_exit_signal(df)  → return df with column 'exit_signal' (1/0)

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
