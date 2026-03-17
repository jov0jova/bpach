"""
Chart data builder for the Charts visualization feature.
Builds Plotly figures from parquet data + user indicator selections.
Covers every column produced by BaseStrategy.populate_indicators().
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Price-overlay indicators (rendered on the candlestick panel) ──────────────

PRICE_OVERLAYS = {
    # EMAs
    "EMA_8":        {"label": "EMA 8",        "color": "#ffa657", "dash": None},
    "EMA_13":       {"label": "EMA 13",       "color": "#e3b341", "dash": None},
    "EMA_20":       {"label": "EMA 20",       "color": "#ff7b72", "dash": None},
    "EMA_50":       {"label": "EMA 50",       "color": "#d2a8ff", "dash": None},
    "EMA_100":      {"label": "EMA 100",      "color": "#a5d6ff", "dash": None},
    "EMA_200":      {"label": "EMA 200",      "color": "#79c0ff", "dash": None},
    # SMAs
    "SMA_8":        {"label": "SMA 8",        "color": "#ffa657", "dash": "dot"},
    "SMA_13":       {"label": "SMA 13",       "color": "#e3b341", "dash": "dot"},
    "SMA_20":       {"label": "SMA 20",       "color": "#ff7b72", "dash": "dot"},
    "SMA_50":       {"label": "SMA 50",       "color": "#d2a8ff", "dash": "dot"},
    # WMAs
    "WMA_9":        {"label": "WMA 9",        "color": "#ffa657", "dash": "dashdot"},
    "WMA_20":       {"label": "WMA 20",       "color": "#e3b341", "dash": "dashdot"},
    # HMAs
    "HMA_9":        {"label": "HMA 9",        "color": "#56d364", "dash": None},
    "HMA_20":       {"label": "HMA 20",       "color": "#3fb950", "dash": None},
    "HMA_50":       {"label": "HMA 50",       "color": "#26a641", "dash": None},
    # TEMA / DEMA
    "TEMA_9":       {"label": "TEMA 9",       "color": "#ff6b6b", "dash": None},
    "TEMA_21":      {"label": "TEMA 21",      "color": "#f94144", "dash": None},
    "DEMA_9":       {"label": "DEMA 9",       "color": "#f3722c", "dash": None},
    "DEMA_21":      {"label": "DEMA 21",      "color": "#f8961e", "dash": None},
    # Adaptive MA
    "KAMA":         {"label": "KAMA",         "color": "#bc8cff", "dash": None},
    # Bollinger Bands
    "BB_upper_20":  {"label": "BB Upper",     "color": "#58a6ff", "dash": "dash"},
    "BB_mid_20":    {"label": "BB Mid",       "color": "#58a6ff", "dash": "dot"},
    "BB_lower_20":  {"label": "BB Lower",     "color": "#58a6ff", "dash": "dash"},
    # Keltner Channels
    "KC_upper":     {"label": "KC Upper",     "color": "#bc8cff", "dash": "dash"},
    "KC_middle":    {"label": "KC Mid",       "color": "#bc8cff", "dash": "dot"},
    "KC_lower":     {"label": "KC Lower",     "color": "#bc8cff", "dash": "dash"},
    # Donchian Channels
    "DC_upper":     {"label": "DC Upper",     "color": "#79c0ff", "dash": "dash"},
    "DC_middle":    {"label": "DC Mid",       "color": "#79c0ff", "dash": "dot"},
    "DC_lower":     {"label": "DC Lower",     "color": "#79c0ff", "dash": "dash"},
    # Momentum / trend lines
    "VWAP":         {"label": "VWAP",         "color": "#f0e68c", "dash": None},
    "PSAR_up":      {"label": "PSAR ↑",       "color": "#3fb950", "dash": None, "markers": True},
    "PSAR_down":    {"label": "PSAR ↓",       "color": "#f85149", "dash": None, "markers": True},
    "SUPERT_10_3":  {"label": "Supertrend",   "color": "#ff9500", "dash": None},
    # Ichimoku
    "ICH_tenkan":   {"label": "Tenkan",       "color": "#ef233c", "dash": None},
    "ICH_kijun":    {"label": "Kijun",        "color": "#4895ef", "dash": None},
    "ICH_senkou_a": {"label": "Senkou A",     "color": "#2dc653", "dash": "dash"},
    "ICH_senkou_b": {"label": "Senkou B",     "color": "#ef233c", "dash": "dash"},
    # Pivot Points
    "PP":           {"label": "Pivot (PP)",   "color": "#f0e68c", "dash": "dot"},
    "R1":           {"label": "R1",           "color": "#f85149", "dash": "dot"},
    "R2":           {"label": "R2",           "color": "#f85149", "dash": "dash"},
    "S1":           {"label": "S1",           "color": "#3fb950", "dash": "dot"},
    "S2":           {"label": "S2",           "color": "#3fb950", "dash": "dash"},
}

# ── Candle-pattern markers (rendered on price panel as arrow symbols) ─────────

CANDLE_PATTERNS = {
    # Single-candle – neutral
    "CDL_DOJI":           {"label": "Doji",             "type": "neutral"},
    "CDL_SPINNING_TOP":   {"label": "Spinning Top",     "type": "neutral"},
    "CDL_INSIDE_BAR":     {"label": "Inside Bar",       "type": "neutral"},
    "CDL_OUTSIDE_BAR":    {"label": "Outside Bar",      "type": "neutral"},
    # Single-candle – bullish
    "CDL_BULL_MARUBOZU":  {"label": "Bull Marubozu",    "type": "bull"},
    "CDL_HAMMER":         {"label": "Hammer",           "type": "bull"},
    "CDL_INV_HAMMER":     {"label": "Inv. Hammer",      "type": "bull"},
    "CDL_DRAGONFLY":      {"label": "Dragonfly Doji",   "type": "bull"},
    # Single-candle – bearish
    "CDL_BEAR_MARUBOZU":  {"label": "Bear Marubozu",    "type": "bear"},
    "CDL_HANGING_MAN":    {"label": "Hanging Man",      "type": "bear"},
    "CDL_SHOOTING_STAR":  {"label": "Shooting Star",    "type": "bear"},
    "CDL_GRAVESTONE":     {"label": "Gravestone Doji",  "type": "bear"},
    # Two-candle – bullish
    "CDL_BULL_ENGULFING": {"label": "Bull Engulfing",   "type": "bull"},
    "CDL_BULL_HARAMI":    {"label": "Bull Harami",      "type": "bull"},
    "CDL_PIERCING":       {"label": "Piercing Line",    "type": "bull"},
    "CDL_TWEEZER_BOTTOM": {"label": "Tweezer Bottom",   "type": "bull"},
    # Two-candle – bearish
    "CDL_BEAR_ENGULFING": {"label": "Bear Engulfing",   "type": "bear"},
    "CDL_BEAR_HARAMI":    {"label": "Bear Harami",      "type": "bear"},
    "CDL_DARK_CLOUD":     {"label": "Dark Cloud Cover", "type": "bear"},
    "CDL_TWEEZER_TOP":    {"label": "Tweezer Top",      "type": "bear"},
    # Three-candle – bullish
    "CDL_3_WHITE_SOLDIERS": {"label": "3 White Soldiers","type": "bull"},
    "CDL_MORNING_STAR":   {"label": "Morning Star",     "type": "bull"},
    "CDL_3_INSIDE_UP":    {"label": "3 Inside Up",      "type": "bull"},
    # Three-candle – bearish
    "CDL_3_BLACK_CROWS":  {"label": "3 Black Crows",    "type": "bear"},
    "CDL_EVENING_STAR":   {"label": "Evening Star",     "type": "bear"},
    "CDL_3_INSIDE_DOWN":  {"label": "3 Inside Down",    "type": "bear"},
}

# ── Sub-pane indicators ───────────────────────────────────────────────────────

PANE_INDICATORS = {
    # Volume
    "volume":         {"label": "Volume",         "pane": "volume",     "color": "#58a6ff",  "bar": True},
    "OBV":            {"label": "OBV",             "pane": "volume",     "color": "#3fb950"},
    "OBV_EMA_12":     {"label": "OBV EMA 12",      "pane": "volume",     "color": "#56d364",  "dash": "dot"},
    "volume_ratio":   {"label": "Vol Ratio",       "pane": "volume",     "color": "#e3b341"},
    "CMF_20":         {"label": "CMF 20",          "pane": "volume",     "color": "#79c0ff"},
    "MFI_14":         {"label": "MFI 14",          "pane": "volume",     "color": "#d2a8ff"},
    "FI_13":          {"label": "Force Index 13",  "pane": "volume",     "color": "#f85149",  "bar": True, "signed_color": True},
    "EOM_14":         {"label": "Ease of Move 14", "pane": "volume",     "color": "#ffa657"},
    "CLOSE_VS_VWAP":  {"label": "Close vs VWAP %", "pane": "volume",     "color": "#f0e68c"},
    # RSI
    "RSI_7":          {"label": "RSI 7",           "pane": "rsi",        "color": "#ff7b72"},
    "RSI_14":         {"label": "RSI 14",          "pane": "rsi",        "color": "#d2a8ff"},
    "RSI_21":         {"label": "RSI 21",          "pane": "rsi",        "color": "#79c0ff"},
    # MACD
    "MACD":           {"label": "MACD",            "pane": "macd",       "color": "#79c0ff"},
    "MACD_signal":    {"label": "Signal",          "pane": "macd",       "color": "#ff7b72"},
    "MACD_hist":      {"label": "Histogram",       "pane": "macd",       "color": "#3fb950",  "bar": True, "signed_color": True},
    # PPO
    "PPO":            {"label": "PPO",             "pane": "ppo",        "color": "#79c0ff"},
    "PPO_signal":     {"label": "PPO Signal",      "pane": "ppo",        "color": "#ff7b72"},
    "PPO_hist":       {"label": "PPO Hist",        "pane": "ppo",        "color": "#3fb950",  "bar": True, "signed_color": True},
    # Stochastic
    "STOCH_K":        {"label": "Stoch %K",        "pane": "stoch",      "color": "#ffa657"},
    "STOCH_D":        {"label": "Stoch %D",        "pane": "stoch",      "color": "#e3b341",  "dash": "dot"},
    "STOCHRSI_K":     {"label": "StochRSI %K",     "pane": "stoch",      "color": "#d2a8ff"},
    "STOCHRSI_D":     {"label": "StochRSI %D",     "pane": "stoch",      "color": "#bc8cff",  "dash": "dot"},
    # ADX / DI / Trend strength
    "ADX_14":         {"label": "ADX 14",          "pane": "adx",        "color": "#f0e68c"},
    "DMP_14":         {"label": "DI+",             "pane": "adx",        "color": "#3fb950"},
    "DMN_14":         {"label": "DI−",             "pane": "adx",        "color": "#f85149"},
    "AROON_up":       {"label": "Aroon Up",        "pane": "adx",        "color": "#56d364"},
    "AROON_down":     {"label": "Aroon Down",      "pane": "adx",        "color": "#f85149",  "dash": "dot"},
    "AROON_osc":      {"label": "Aroon Osc",       "pane": "adx",        "color": "#e3b341",  "bar": True, "signed_color": True},
    "VI_pos":         {"label": "Vortex +",        "pane": "adx",        "color": "#3fb950",  "dash": "dash"},
    "VI_neg":         {"label": "Vortex −",        "pane": "adx",        "color": "#f85149",  "dash": "dash"},
    # Momentum
    "AO":             {"label": "Awesome Osc",     "pane": "momentum",   "color": "#3fb950",  "bar": True, "signed_color": True},
    "ROC_10":         {"label": "ROC 10",          "pane": "momentum",   "color": "#ffa657"},
    "ROC_20":         {"label": "ROC 20",          "pane": "momentum",   "color": "#e3b341",  "dash": "dot"},
    "CMO_14":         {"label": "CMO 14",          "pane": "momentum",   "color": "#bc8cff"},
    "FISHER":         {"label": "Fisher Transform","pane": "momentum",   "color": "#79c0ff"},
    "UO":             {"label": "Ultimate Osc",    "pane": "momentum",   "color": "#58a6ff"},
    "KST":            {"label": "KST",             "pane": "momentum",   "color": "#d2a8ff"},
    "KST_signal":     {"label": "KST Signal",      "pane": "momentum",   "color": "#f85149",  "dash": "dot"},
    "TRIX_15":        {"label": "TRIX 15",         "pane": "momentum",   "color": "#3fb950"},
    "DPO_20":         {"label": "DPO 20",          "pane": "momentum",   "color": "#ffa657"},
    # Classic oscillators
    "CCI_20":         {"label": "CCI 20",          "pane": "misc",       "color": "#bc8cff"},
    "WILLR_14":       {"label": "Williams %R",     "pane": "misc",       "color": "#58a6ff"},
    # Volatility
    "ATR_14":         {"label": "ATR 14",          "pane": "volatility", "color": "#ffa657"},
    "NATR_14":        {"label": "NATR %",          "pane": "volatility", "color": "#e3b341"},
    "BB_width_20":    {"label": "BB Width",        "pane": "volatility", "color": "#58a6ff"},
    "BB_pct_20":      {"label": "BB %B",           "pane": "volatility", "color": "#d2a8ff"},
    "HV_20":          {"label": "Hist. Volatility","pane": "volatility", "color": "#79c0ff"},
    "DC_WIDTH":       {"label": "Donchian Width %","pane": "volatility", "color": "#a5d6ff"},
    "KC_WIDTH":       {"label": "Keltner Width %", "pane": "volatility", "color": "#bc8cff"},
    "BB_SQUEEZE":     {"label": "BB Squeeze",      "pane": "volatility", "color": "#f0e68c",  "bar": True},
    # Candle metrics & price change
    "close_pct_change": {"label": "Close Chg %",  "pane": "candle",     "color": "#58a6ff",  "bar": True, "signed_color": True},
    "body_pct":         {"label": "Body %",        "pane": "candle",     "color": "#ffa657"},
    "upper_wick_pct":   {"label": "Upper Wick %",  "pane": "candle",     "color": "#f85149"},
    "lower_wick_pct":   {"label": "Lower Wick %",  "pane": "candle",     "color": "#3fb950"},
    "PRICE_RANGE_PCT":  {"label": "Price Range %", "pane": "candle",     "color": "#d2a8ff"},
    "EMA50_slope":      {"label": "EMA 50 Slope",  "pane": "candle",     "color": "#e3b341",  "bar": True, "signed_color": True},
}

PANE_ORDER = [
    "volume", "rsi", "macd", "ppo", "stoch",
    "adx", "momentum", "misc", "volatility", "candle",
]

PANE_HEIGHT_SHARE = {
    "volume":     0.08,
    "rsi":        0.10,
    "macd":       0.10,
    "ppo":        0.10,
    "stoch":      0.10,
    "adx":        0.10,
    "momentum":   0.10,
    "misc":       0.10,
    "volatility": 0.10,
    "candle":     0.10,
}

PANE_HLINES = {
    "rsi":      [(70, "#f85149", "dot"), (50, "#586069", "dot"), (30, "#3fb950", "dot")],
    "stoch":    [(80, "#f85149", "dot"), (20, "#3fb950", "dot")],
    "misc":     [(0, "#586069", "dot"), (-100, "#f85149", "dot"), (100, "#3fb950", "dot")],
    "adx":      [(25, "#f0e68c", "dot")],
    "momentum": [(0, "#586069", "dot")],
    "ppo":      [(0, "#586069", "dot")],
    "candle":   [(0, "#586069", "dot")],
    "volatility": [],
}

_DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", size=11),
    legend=dict(
        bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d", borderwidth=1,
        font=dict(size=10), orientation="h",
        yanchor="bottom", y=1.01, xanchor="left", x=0,
    ),
    margin=dict(l=60, r=12, t=36, b=28),
    hovermode="x unified",
)


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    selected_overlays: list,
    selected_panes: list,
    selected_patterns: list,
    candle_limit: int = 500,
) -> str:
    """
    Build a multi-pane candlestick chart for one symbol.
    Returns a Plotly JSON string ready for Plotly.react() / Plotly.newPlot().
    """
    if len(df) > candle_limit:
        df = df.tail(candle_limit).reset_index(drop=True)

    # Determine which pane groups are actually needed
    needed_panes: list[str] = []
    for pane_id in PANE_ORDER:
        for ind in selected_panes:
            spec = PANE_INDICATORS.get(ind)
            if spec and spec["pane"] == pane_id and pane_id not in needed_panes:
                needed_panes.append(pane_id)
                break

    total_rows = 1 + len(needed_panes)
    pane_heights_raw = [PANE_HEIGHT_SHARE.get(p, 0.10) for p in needed_panes]
    price_height = max(0.38, 1.0 - sum(pane_heights_raw))
    raw_heights = [price_height] + pane_heights_raw
    total_h = sum(raw_heights)
    row_heights = [h / total_h for h in raw_heights]
    pane_row = {pane: idx + 2 for idx, pane in enumerate(needed_panes)}

    subplot_titles = [f"<b>{symbol}</b>  {timeframe}"] + [p.upper() for p in needed_panes]

    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.012,
        subplot_titles=subplot_titles,
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing=dict(line=dict(color="#3fb950"), fillcolor="#3fb950"),
        decreasing=dict(line=dict(color="#f85149"), fillcolor="#f85149"),
        showlegend=False,
        whiskerwidth=0.4,
    ), row=1, col=1)

    # ── Ichimoku cloud fill (needs both senkou lines) ─────────────────────────
    has_cloud = (
        "ICH_senkou_a" in selected_overlays and "ICH_senkou_b" in selected_overlays
        and "ICH_senkou_a" in df.columns and "ICH_senkou_b" in df.columns
    )
    if has_cloud:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["ICH_senkou_a"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["ICH_senkou_b"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(88,166,255,0.07)",
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # ── Price overlays ────────────────────────────────────────────────────────
    for ind in selected_overlays:
        spec = PRICE_OVERLAYS.get(ind)
        if spec is None or ind not in df.columns:
            continue
        if spec.get("markers"):
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[ind],
                mode="markers",
                marker=dict(symbol="circle", size=3, color=spec["color"]),
                name=spec["label"],
            ), row=1, col=1)
        else:
            line_cfg = dict(color=spec["color"], width=1)
            if spec.get("dash"):
                line_cfg["dash"] = spec["dash"]
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[ind],
                mode="lines", line=line_cfg,
                name=spec["label"], opacity=0.9,
            ), row=1, col=1)

    # ── Candle pattern markers ────────────────────────────────────────────────
    for pat in selected_patterns:
        spec = CANDLE_PATTERNS.get(pat)
        if spec is None or pat not in df.columns:
            continue
        ptype = spec["type"]
        if ptype == "neutral":
            mask = df[pat] != 0
            if mask.any():
                mid = (df["high"] + df["low"]) / 2
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, "timestamp"],
                    y=mid[mask],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=8, color="#f0e68c",
                                line=dict(width=1.5, color="#f0e68c")),
                    name=spec["label"],
                ), row=1, col=1)
        elif ptype == "bull":
            mask = df[pat] > 0
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, "timestamp"],
                    y=df.loc[mask, "low"] * 0.9985,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=9, color="#3fb950"),
                    name=spec["label"],
                ), row=1, col=1)
        elif ptype == "bear":
            mask = df[pat] < 0
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, "timestamp"],
                    y=df.loc[mask, "high"] * 1.0015,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=9, color="#f85149"),
                    name=spec["label"],
                ), row=1, col=1)

    # ── Sub-pane indicators ───────────────────────────────────────────────────
    for ind in selected_panes:
        spec = PANE_INDICATORS.get(ind)
        if spec is None or ind not in df.columns:
            continue
        pane_id = spec["pane"]
        row = pane_row.get(pane_id)
        if row is None:
            continue

        if spec.get("bar"):
            if spec.get("signed_color"):
                colors = ["#3fb950" if v >= 0 else "#f85149" for v in df[ind]]
            elif ind == "volume":
                colors = ["#3fb950" if c >= o else "#f85149"
                          for c, o in zip(df["close"], df["open"])]
            else:
                colors = spec["color"]
            fig.add_trace(go.Bar(
                x=df["timestamp"], y=df[ind],
                name=spec["label"],
                marker_color=colors,
                showlegend=(ind != "volume"),
            ), row=row, col=1)
        else:
            line_cfg = dict(color=spec["color"], width=1)
            if spec.get("dash"):
                line_cfg["dash"] = spec["dash"]
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[ind],
                mode="lines", line=line_cfg,
                name=spec["label"],
            ), row=row, col=1)

    # ── Reference lines ───────────────────────────────────────────────────────
    for pane_id, hlines in PANE_HLINES.items():
        row = pane_row.get(pane_id)
        if row is None:
            continue
        for level, color, dash in hlines:
            fig.add_hline(
                y=level, row=row, col=1,
                line=dict(color=color, width=1, dash=dash),
                opacity=0.40,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    chart_height = max(420, 300 + len(needed_panes) * 90)
    layout = dict(**_DARK, height=chart_height)

    axis_style = dict(
        gridcolor="#21262d", linecolor="#30363d",
        showspikes=True, spikecolor="#58a6ff", spikethickness=1,
        rangeslider=dict(visible=False),
    )
    for i in range(1, total_rows + 1):
        suffix = "" if i == 1 else str(i)
        layout[f"xaxis{suffix}"] = dict(**axis_style)
        layout[f"yaxis{suffix}"] = dict(
            gridcolor="#21262d", linecolor="#30363d",
            showspikes=True, spikecolor="#58a6ff", spikethickness=1,
        )

    fig.update_layout(**layout)
    fig.update_layout(xaxis_rangeslider_visible=False, dragmode="pan")
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#8b949e")

    return fig.to_json()


# ── Grouped indicator lists for the UI ───────────────────────────────────────

OVERLAY_GROUPS = {
    "EMAs": [
        "EMA_8", "EMA_13", "EMA_20", "EMA_50", "EMA_100", "EMA_200",
    ],
    "SMAs / WMAs": [
        "SMA_8", "SMA_13", "SMA_20", "SMA_50",
        "WMA_9", "WMA_20",
    ],
    "HMA / TEMA / DEMA": [
        "HMA_9", "HMA_20", "HMA_50",
        "TEMA_9", "TEMA_21", "DEMA_9", "DEMA_21", "KAMA",
    ],
    "Bands & Channels": [
        "BB_upper_20", "BB_mid_20", "BB_lower_20",
        "KC_upper", "KC_middle", "KC_lower",
        "DC_upper", "DC_middle", "DC_lower",
    ],
    "Trend Lines": [
        "VWAP", "PSAR_up", "PSAR_down", "SUPERT_10_3",
        "ICH_tenkan", "ICH_kijun", "ICH_senkou_a", "ICH_senkou_b",
    ],
    "Pivot Points": ["PP", "R1", "R2", "S1", "S2"],
}

CANDLE_PATTERN_GROUPS = {
    "Single-Candle": [
        "CDL_DOJI", "CDL_SPINNING_TOP", "CDL_INSIDE_BAR", "CDL_OUTSIDE_BAR",
        "CDL_BULL_MARUBOZU", "CDL_BEAR_MARUBOZU",
        "CDL_HAMMER", "CDL_INV_HAMMER", "CDL_HANGING_MAN", "CDL_SHOOTING_STAR",
        "CDL_DRAGONFLY", "CDL_GRAVESTONE",
    ],
    "Two-Candle": [
        "CDL_BULL_ENGULFING", "CDL_BEAR_ENGULFING",
        "CDL_BULL_HARAMI", "CDL_BEAR_HARAMI",
        "CDL_PIERCING", "CDL_DARK_CLOUD",
        "CDL_TWEEZER_BOTTOM", "CDL_TWEEZER_TOP",
    ],
    "Three-Candle": [
        "CDL_3_WHITE_SOLDIERS", "CDL_3_BLACK_CROWS",
        "CDL_MORNING_STAR", "CDL_EVENING_STAR",
        "CDL_3_INSIDE_UP", "CDL_3_INSIDE_DOWN",
    ],
}

PANE_GROUPS = {
    "Volume & Money Flow": [
        "volume", "OBV", "OBV_EMA_12", "volume_ratio",
        "CMF_20", "MFI_14", "FI_13", "EOM_14", "CLOSE_VS_VWAP",
    ],
    "RSI":          ["RSI_7", "RSI_14", "RSI_21"],
    "MACD":         ["MACD", "MACD_signal", "MACD_hist"],
    "PPO":          ["PPO", "PPO_signal", "PPO_hist"],
    "Stochastic":   ["STOCH_K", "STOCH_D", "STOCHRSI_K", "STOCHRSI_D"],
    "ADX / Aroon / Vortex": [
        "ADX_14", "DMP_14", "DMN_14",
        "AROON_up", "AROON_down", "AROON_osc",
        "VI_pos", "VI_neg",
    ],
    "Momentum": [
        "AO", "ROC_10", "ROC_20", "CMO_14", "FISHER",
        "UO", "KST", "KST_signal", "TRIX_15", "DPO_20",
    ],
    "Oscillators":  ["CCI_20", "WILLR_14"],
    "Volatility": [
        "ATR_14", "NATR_14", "BB_width_20", "BB_pct_20",
        "HV_20", "DC_WIDTH", "KC_WIDTH", "BB_SQUEEZE",
    ],
    "Candle Metrics & Price": [
        "close_pct_change", "body_pct",
        "upper_wick_pct", "lower_wick_pct",
        "PRICE_RANGE_PCT", "EMA50_slope",
    ],
}

# Defaults shown when first opening Charts
DEFAULT_OVERLAYS  = ["EMA_20", "EMA_50", "EMA_200"]
DEFAULT_PANES     = ["volume", "RSI_14", "MACD", "MACD_signal", "MACD_hist"]
DEFAULT_PATTERNS  = []
