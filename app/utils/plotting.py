"""
Plotly chart builders for the analysis and backtest views.
All charts use a consistent dark theme matching the app.
"""
import json
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", size=12),
    xaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        showspikes=True,
        spikecolor="#58a6ff",
        spikethickness=1,
    ),
    yaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        showspikes=True,
        spikecolor="#58a6ff",
        spikethickness=1,
    ),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode="x unified",
)


def candlestick_chart(df: pd.DataFrame, symbol: str, timeframe: str,
                      entry_times=None, exit_times=None,
                      indicators: Optional[list] = None) -> str:
    """
    Build a candlestick chart with optional entry/exit markers and indicator overlays.
    Returns JSON string for Plotly.
    """
    row_heights = [0.6, 0.2, 0.2]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=row_heights,
        vertical_spacing=0.02,
        subplot_titles=[f"{symbol} {timeframe}", "Volume", "RSI"]
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing=dict(line=dict(color="#3fb950"), fillcolor="#3fb950"),
        decreasing=dict(line=dict(color="#f85149"), fillcolor="#f85149"),
    ), row=1, col=1)

    # EMA overlays if present
    for col_name, color, label in [
        ("EMA_20", "#ffa657", "EMA 20"),
        ("EMA_50", "#d2a8ff", "EMA 50"),
        ("EMA_200", "#79c0ff", "EMA 200"),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[col_name],
                mode="lines", line=dict(color=color, width=1),
                name=label, opacity=0.8
            ), row=1, col=1)

    # Entry markers
    if entry_times is not None and len(entry_times) > 0:
        entry_prices = []
        for et in entry_times:
            row = df[df["timestamp"] == pd.Timestamp(et)]
            if not row.empty:
                entry_prices.append(float(row["low"].iloc[0]) * 0.998)
            else:
                entry_prices.append(None)
        fig.add_trace(go.Scatter(
            x=list(entry_times),
            y=entry_prices,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#3fb950"),
            name="Entry",
        ), row=1, col=1)

    # Exit markers
    if exit_times is not None and len(exit_times) > 0:
        exit_prices = []
        for et in exit_times:
            row = df[df["timestamp"] == pd.Timestamp(et)]
            if not row.empty:
                exit_prices.append(float(row["high"].iloc[0]) * 1.002)
            else:
                exit_prices.append(None)
        fig.add_trace(go.Scatter(
            x=list(exit_times),
            y=exit_prices,
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#f85149"),
            name="Exit",
        ), row=1, col=1)

    # Volume
    colors = ["#3fb950" if c >= o else "#f85149"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["volume"],
        marker_color=colors, name="Volume", opacity=0.7
    ), row=2, col=1)

    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["RSI_14"],
            mode="lines", line=dict(color="#79c0ff", width=1.5),
            name="RSI 14"
        ), row=3, col=1)
        for level, color in [(30, "#3fb950"), (50, "#8b949e"), (70, "#f85149")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                          opacity=0.5, row=3, col=1)

    layout = {**DARK_LAYOUT,
              "title": {"text": f"{symbol} — {timeframe}", "font": {"size": 14}},
              "xaxis3": {"rangeslider": {"visible": False}},
              "height": 700}
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")

    return fig.to_json()


def equity_curve_chart(equity: list, timestamps: list, title: str = "Equity Curve") -> str:
    """Line chart of portfolio equity over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=equity,
        mode="lines", fill="tozeroy",
        line=dict(color="#58a6ff", width=2),
        fillcolor="rgba(88,166,255,0.1)",
        name="Equity"
    ))
    fig.update_layout(**DARK_LAYOUT, title=title, height=350)
    return fig.to_json()


def indicator_heatmap(corr_data: dict, title: str = "Indicator Correlation") -> str:
    """Correlation heatmap of indicator values at winner entries."""
    import numpy as np
    indicators = list(corr_data.keys())
    if not indicators:
        return "{}"
    n = len(indicators)
    matrix = [[corr_data.get(row, {}).get(col, 0) for col in indicators] for row in indicators]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=indicators, y=indicators,
        colorscale="RdBu", zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        colorbar=dict(tickfont=dict(color="#c9d1d9"))
    ))
    fig.update_layout(**DARK_LAYOUT, title=title, height=500)
    return fig.to_json()


def win_rate_bar_chart(data: dict, title: str = "Win Rate by Condition") -> str:
    """Horizontal bar chart: condition label → win rate."""
    labels = list(data.keys())
    values = [data[l] * 100 for l in labels]
    colors = ["#3fb950" if v >= 60 else "#ffa657" if v >= 50 else "#f85149"
              for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in values],
        textposition="outside"
    ))
    fig.update_layout(**DARK_LAYOUT, title=title, height=max(300, len(labels) * 25 + 100),
                      xaxis=dict(range=[0, 105], title="Win Rate (%)"))
    return fig.to_json()


def pnl_distribution(pnl_values: list, title: str = "PnL Distribution") -> str:
    """Histogram of PnL percentages."""
    fig = go.Figure(go.Histogram(
        x=pnl_values, nbinsx=50,
        marker_color="#58a6ff", opacity=0.8,
        marker_line=dict(color="#21262d", width=0.5)
    ))
    fig.add_vline(x=0, line_color="#8b949e", line_dash="dash")
    fig.update_layout(**DARK_LAYOUT, title=title, height=300,
                      xaxis=dict(title="PnL %"), yaxis=dict(title="Count"))
    return fig.to_json()
