"""
Chart data builder for the Charts visualization feature.
Builds Plotly figures from parquet data + user indicator selections.
"""
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Indicator definitions ─────────────────────────────────────────────────────

# Rendered as line/marker overlays on the price (candlestick) panel
PRICE_OVERLAYS = {
    "EMA_8":        {"label": "EMA 8",       "color": "#ffa657", "dash": None},
    "EMA_13":       {"label": "EMA 13",      "color": "#e3b341", "dash": None},
    "EMA_20":       {"label": "EMA 20",      "color": "#ff7b72", "dash": None},
    "EMA_50":       {"label": "EMA 50",      "color": "#d2a8ff", "dash": None},
    "EMA_100":      {"label": "EMA 100",     "color": "#a5d6ff", "dash": None},
    "EMA_200":      {"label": "EMA 200",     "color": "#79c0ff", "dash": None},
    "SMA_8":        {"label": "SMA 8",       "color": "#ffa657", "dash": "dot"},
    "SMA_13":       {"label": "SMA 13",      "color": "#e3b341", "dash": "dot"},
    "SMA_20":       {"label": "SMA 20",      "color": "#ff7b72", "dash": "dot"},
    "SMA_50":       {"label": "SMA 50",      "color": "#d2a8ff", "dash": "dot"},
    "HMA_9":        {"label": "HMA 9",       "color": "#56d364", "dash": None},
    "HMA_20":       {"label": "HMA 20",      "color": "#3fb950", "dash": None},
    "HMA_50":       {"label": "HMA 50",      "color": "#26a641", "dash": None},
    "TEMA_9":       {"label": "TEMA 9",      "color": "#ff6b6b", "dash": None},
    "TEMA_21":      {"label": "TEMA 21",     "color": "#f94144", "dash": None},
    "DEMA_9":       {"label": "DEMA 9",      "color": "#f3722c", "dash": None},
    "DEMA_21":      {"label": "DEMA 21",     "color": "#f8961e", "dash": None},
    "BB_upper_20":  {"label": "BB Upper",    "color": "#58a6ff", "dash": "dash"},
    "BB_lower_20":  {"label": "BB Lower",    "color": "#58a6ff", "dash": "dash"},
    "KC_upper":     {"label": "KC Upper",    "color": "#bc8cff", "dash": "dash"},
    "KC_lower":     {"label": "KC Lower",    "color": "#bc8cff", "dash": "dash"},
    "VWAP":         {"label": "VWAP",        "color": "#f0e68c", "dash": None},
    "PSAR":         {"label": "PSAR",        "color": "#ff9500", "dash": None, "markers": True},
    "ICH_tenkan":   {"label": "Tenkan",      "color": "#ef233c", "dash": None},
    "ICH_kijun":    {"label": "Kijun",       "color": "#4895ef", "dash": None},
    "ICH_senkou_a": {"label": "Senkou A",    "color": "#2dc653", "dash": "dash"},
    "ICH_senkou_b": {"label": "Senkou B",    "color": "#ef233c", "dash": "dash"},
}

# Rendered in separate sub-panels below the price chart
PANE_INDICATORS = {
    # Volume
    "volume":       {"label": "Volume",       "pane": "volume",     "color": "#58a6ff",  "bar": True},
    "OBV":          {"label": "OBV",           "pane": "volume",     "color": "#3fb950"},
    "volume_ratio": {"label": "Vol Ratio",     "pane": "volume",     "color": "#e3b341"},
    # RSI
    "RSI_7":        {"label": "RSI 7",         "pane": "rsi",        "color": "#ff7b72"},
    "RSI_14":       {"label": "RSI 14",        "pane": "rsi",        "color": "#d2a8ff"},
    "RSI_21":       {"label": "RSI 21",        "pane": "rsi",        "color": "#79c0ff"},
    # MACD
    "MACD":         {"label": "MACD",          "pane": "macd",       "color": "#79c0ff"},
    "MACD_signal":  {"label": "Signal",        "pane": "macd",       "color": "#ff7b72"},
    "MACD_hist":    {"label": "Histogram",     "pane": "macd",       "color": "#3fb950",  "bar": True, "signed_color": True},
    # Stochastic
    "STOCH_K":      {"label": "Stoch %K",      "pane": "stoch",      "color": "#ffa657"},
    "STOCHRSI_K":   {"label": "StochRSI",      "pane": "stoch",      "color": "#d2a8ff"},
    # ADX / DI
    "ADX_14":       {"label": "ADX 14",        "pane": "adx",        "color": "#f0e68c"},
    "DMP_14":       {"label": "DI+",           "pane": "adx",        "color": "#3fb950"},
    "DMN_14":       {"label": "DI−",           "pane": "adx",        "color": "#f85149"},
    "AROON_up":     {"label": "Aroon Up",      "pane": "adx",        "color": "#56d364"},
    # Momentum
    "AO":           {"label": "Awesome Osc",   "pane": "momentum",   "color": "#3fb950",  "bar": True, "signed_color": True},
    "CMF_20":       {"label": "CMF 20",        "pane": "momentum",   "color": "#79c0ff"},
    "ROC_10":       {"label": "ROC 10",        "pane": "momentum",   "color": "#ffa657"},
    # Misc oscillators
    "CCI_20":       {"label": "CCI 20",        "pane": "misc",       "color": "#bc8cff"},
    "WILLR_14":     {"label": "Williams %R",   "pane": "misc",       "color": "#58a6ff"},
    "MFI_14":       {"label": "MFI 14",        "pane": "misc",       "color": "#3fb950"},
    # Volatility
    "ATR_14":       {"label": "ATR 14",        "pane": "volatility", "color": "#ffa657"},
    "NATR_14":      {"label": "NATR %",        "pane": "volatility", "color": "#e3b341"},
    "BB_width_20":  {"label": "BB Width",      "pane": "volatility", "color": "#58a6ff"},
    "BB_pct_20":    {"label": "BB %B",         "pane": "volatility", "color": "#d2a8ff"},
}

PANE_ORDER = ["volume", "rsi", "macd", "stoch", "adx", "momentum", "misc", "volatility"]

# Heights as fraction of total (price gets whatever is left)
PANE_HEIGHT_SHARE = {
    "volume":     0.08,
    "rsi":        0.11,
    "macd":       0.11,
    "stoch":      0.10,
    "adx":        0.10,
    "momentum":   0.10,
    "misc":       0.10,
    "volatility": 0.10,
}

# Reference lines drawn on specific panes
PANE_HLINES = {
    "rsi":   [(70, "#f85149", "dot"), (50, "#586069", "dot"), (30, "#3fb950", "dot")],
    "stoch": [(80, "#f85149", "dot"), (20, "#3fb950", "dot")],
    "misc":  [(0,  "#586069", "dot"), (-100, "#f85149", "dot"), (100, "#3fb950", "dot")],
    "adx":   [(25, "#f0e68c", "dot")],
    "momentum": [(0, "#586069", "dot")],
}

_DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", size=11),
    legend=dict(bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
                borderwidth=1, font=dict(size=10),
                orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(l=60, r=12, t=36, b=28),
    hovermode="x unified",
)


def build_chart(df: pd.DataFrame, symbol: str, timeframe: str,
                selected_overlays: list, selected_panes: list,
                candle_limit: int = 500) -> str:
    """
    Build a multi-pane candlestick chart for one symbol.
    Returns a Plotly JSON string ready for Plotly.react() / Plotly.newPlot().
    """
    if len(df) > candle_limit:
        df = df.tail(candle_limit).reset_index(drop=True)

    # Determine which pane groups are actually needed
    needed_panes = []
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

    # ── Price overlays ────────────────────────────────────────────────────────
    # Ichimoku cloud fill (needs both senkou lines)
    has_cloud = ("ICH_senkou_a" in selected_overlays and "ICH_senkou_b" in selected_overlays
                 and "ICH_senkou_a" in df.columns and "ICH_senkou_b" in df.columns)
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
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[ind],
                mode="lines",
                line=dict(color=spec["color"], width=1),
                name=spec["label"],
            ), row=row, col=1)

    # ── Reference lines on panes ──────────────────────────────────────────────
    for pane_id, hlines in PANE_HLINES.items():
        row = pane_row.get(pane_id)
        if row is None:
            continue
        for level, color, dash in hlines:
            fig.add_hline(
                y=level, row=row, col=1,
                line=dict(color=color, width=1, dash=dash),
                opacity=0.45,
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
    # Keep subplot title text small
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#8b949e")

    return fig.to_json()


# ── Grouped indicator lists for the UI ───────────────────────────────────────

OVERLAY_GROUPS = {
    "Moving Averages": ["EMA_8", "EMA_13", "EMA_20", "EMA_50", "EMA_100", "EMA_200",
                        "SMA_8", "SMA_13", "SMA_20", "SMA_50",
                        "HMA_9", "HMA_20", "HMA_50",
                        "TEMA_9", "TEMA_21", "DEMA_9", "DEMA_21"],
    "Bands & Channels": ["BB_upper_20", "BB_lower_20", "KC_upper", "KC_lower"],
    "Trend Lines":      ["VWAP", "PSAR", "ICH_tenkan", "ICH_kijun",
                         "ICH_senkou_a", "ICH_senkou_b"],
}

PANE_GROUPS = {
    "Volume":     ["volume", "OBV", "volume_ratio"],
    "RSI":        ["RSI_7", "RSI_14", "RSI_21"],
    "MACD":       ["MACD", "MACD_signal", "MACD_hist"],
    "Stochastic": ["STOCH_K", "STOCHRSI_K"],
    "ADX / DI":   ["ADX_14", "DMP_14", "DMN_14", "AROON_up"],
    "Momentum":   ["AO", "CMF_20", "ROC_10"],
    "Oscillators":["CCI_20", "WILLR_14", "MFI_14"],
    "Volatility": ["ATR_14", "NATR_14", "BB_width_20", "BB_pct_20"],
}

# Default selections shown when user first opens Charts
DEFAULT_OVERLAYS = ["EMA_20", "EMA_50", "EMA_200"]
DEFAULT_PANES    = ["volume", "RSI_14", "MACD", "MACD_signal", "MACD_hist"]
