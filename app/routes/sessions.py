import json

import ccxt
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..utils import db

bp = Blueprint("sessions", __name__)


@bp.route("/")
def list_sessions():
    sessions = db.list_sessions()
    return render_template("sessions/list.html", sessions=sessions)


@bp.route("/new", methods=["GET", "POST"])
def new_session():
    exchanges = ["binance", "bybit", "okx", "kucoin", "gate", "huobi", "kraken"]
    timeframe_choices = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        exchange = request.form.get("exchange", "binance")
        timeframes = request.form.getlist("timeframes")
        quote_asset = request.form.get("quote_asset", "USDT").upper().strip()

        if not name:
            flash("Session name is required.", "error")
            return render_template("sessions/new.html",
                                   exchanges=exchanges,
                                   timeframe_choices=timeframe_choices)
        if not timeframes:
            timeframes = ["1h"]

        session = db.create_session(name, exchange, timeframes, quote_asset)

        # Pre-fetch pairs list from exchange
        try:
            ex_class = getattr(ccxt, exchange)
            ex = ex_class({"enableRateLimit": True})
            markets = ex.load_markets()
            pairs = []
            for symbol, market in markets.items():
                if not market.get("active", True):
                    continue
                if market.get("quote") != quote_asset:
                    continue
                pairs.append({
                    "symbol": symbol,
                    "base_asset": market.get("base", ""),
                    "quote_asset": quote_asset,
                    "active": True,
                    "volume_24h": 0.0,
                })
            db.upsert_pairs(session["id"], pairs[:current_app.config["MAX_PAIRS"]])
        except Exception as e:
            flash(f"Warning: could not pre-load pairs from {exchange}: {e}", "warning")

        flash(f"Session '{name}' created.", "success")
        return redirect(url_for("sessions.detail", session_id=session["id"]))

    return render_template("sessions/new.html",
                           exchanges=exchanges,
                           timeframe_choices=timeframe_choices)


@bp.route("/<session_id>")
def detail(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    pairs = db.list_pairs(session_id)
    runs = db.list_backtest_runs(session_id)
    algo_results = db.list_algo_results(session_id)

    # Get latest task for each phase
    tasks = {
        "download": db.get_latest_task(session_id, "download"),
        "indicators": db.get_latest_task(session_id, "indicators"),
        "backtest": db.get_latest_task(session_id, "backtest"),
        "analysis": db.get_latest_task(session_id, "analysis"),
        "algofinder": db.get_latest_task(session_id, "algofinder"),
        "path_a_analysis": db.get_latest_task(session_id, "path_a_analysis"),
    }

    return render_template("sessions/detail.html",
                           session=session,
                           pairs=pairs,
                           runs=runs,
                           algo_results=algo_results,
                           tasks=tasks,
                           pair_count=len(pairs),
                           active_pair_count=sum(1 for p in pairs if not p["excluded"]))


@bp.route("/<session_id>/delete", methods=["POST"])
def delete_session(session_id):
    session = db.get_session(session_id)
    if session:
        db.delete_session(session_id)
        flash(f"Session '{session['name']}' deleted.", "success")
    return redirect(url_for("sessions.list_sessions"))


@bp.route("/tasks/<task_id>/status")
def task_status(task_id):
    """HTMX polling endpoint – returns a progress bar fragment."""
    task = db.get_task(task_id)
    if not task:
        return "<div>Task not found</div>", 404
    return render_template("partials/task_progress.html", task=task)
