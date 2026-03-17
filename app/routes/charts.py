"""
Charts blueprint – multi-coin, multi-timeframe visualization engine.
"""
import json

import pandas as pd
from flask import Blueprint, abort, jsonify, render_template, request

import app.models as m
from app.config import Config
from app.services.charts import (
    DEFAULT_OVERLAYS, DEFAULT_PANES,
    OVERLAY_GROUPS, PANE_GROUPS,
    PRICE_OVERLAYS, PANE_INDICATORS,
    build_chart,
)
from app.utils.parquet import parquet_path

bp = Blueprint("charts", __name__)


def _session_or_404(session_id):
    s = m.get_session(Config.DB_PATH, session_id)
    if not s:
        abort(404)
    return s


@bp.get("/<session_id>")
def charts_view(session_id):
    session = _session_or_404(session_id)
    pairs = m.list_pairs(Config.DB_PATH, session_id)
    active_pairs = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    layouts = m.list_chart_layouts(Config.DB_PATH, session_id)

    return render_template(
        "charts/view.html",
        session=session,
        pairs=active_pairs,
        timeframes=session.get("timeframes") or ["1h"],
        overlay_groups=OVERLAY_GROUPS,
        pane_groups=PANE_GROUPS,
        price_overlays=PRICE_OVERLAYS,
        pane_indicators=PANE_INDICATORS,
        default_overlays=DEFAULT_OVERLAYS,
        default_panes=DEFAULT_PANES,
        layouts=layouts,
    )


@bp.post("/<session_id>/chart")
def get_chart(session_id):
    """Return Plotly JSON for a single pair."""
    _session_or_404(session_id)
    data = request.get_json(silent=True) or {}

    symbol       = data.get("symbol", "")
    timeframe    = data.get("timeframe", "1h")
    overlays     = data.get("overlays", [])
    panes        = data.get("panes", [])
    candle_limit = int(data.get("candle_limit", 500))

    path = parquet_path(Config.PARQUET_DIR, session_id, symbol, timeframe)
    if not path.exists():
        return jsonify({"error": f"No data for {symbol} {timeframe}. Run Download + Add Indicators first."}), 404

    df = pd.read_parquet(path)
    if df.empty:
        return jsonify({"error": f"{symbol}: dataset is empty"}), 404

    chart_json = build_chart(df, symbol, timeframe, overlays, panes, candle_limit)
    return jsonify({"chart": chart_json, "symbol": symbol, "candles": len(df)})


@bp.post("/<session_id>/layout/save")
def save_layout(session_id):
    _session_or_404(session_id)
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Layout name is required"}), 400
    config = data.get("config", {})
    layout_id = m.save_chart_layout(Config.DB_PATH, session_id, name, config)
    return jsonify({"id": layout_id, "name": name})


@bp.post("/<session_id>/layout/<layout_id>/delete")
def delete_layout(session_id, layout_id):
    _session_or_404(session_id)
    m.delete_chart_layout(Config.DB_PATH, layout_id, session_id)
    return jsonify({"ok": True})


@bp.get("/<session_id>/layouts")
def list_layouts(session_id):
    _session_or_404(session_id)
    layouts = m.list_chart_layouts(Config.DB_PATH, session_id)
    return jsonify({"layouts": layouts})
