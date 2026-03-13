from flask import Blueprint, current_app, flash, redirect, render_template, url_for

from ..tasks.runner import submit_task, request_cancel
from ..services.indicators import run_indicators, run_reset_indicators
from ..utils import db

bp = Blueprint("indicators", __name__)


@bp.route("/<session_id>")
def indicators_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "indicators")
    pairs = db.list_pairs(session_id)
    ready = sum(1 for p in pairs if p["candle_count"] > 0)

    return render_template("indicators/view.html",
                           session=session,
                           task=task,
                           total_pairs=len(pairs),
                           ready_pairs=ready)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task_id = submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "indicators",
        run_indicators,
        session_id,
        current_app.config["PARQUET_DIR"],
        session["timeframes"],
    )

    flash("Indicator enrichment started.", "info")
    return redirect(url_for("indicators.indicators_view", session_id=session_id))


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "indicators")
    return render_template("partials/task_progress.html", task=task)


@bp.route("/<session_id>/cancel", methods=["POST"])
def cancel(session_id):
    task = db.get_latest_task(session_id, "indicators")
    if task and task["status"] == "running":
        request_cancel(current_app.config["DB_PATH"], task["id"])
        flash("Stop signal sent — current jobs will finish then the task will halt.", "warning")
    return redirect(url_for("indicators.indicators_view", session_id=session_id))


@bp.route("/<session_id>/reset", methods=["POST"])
def reset(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "indicators",
        run_reset_indicators,
        session_id,
        current_app.config["PARQUET_DIR"],
        session["timeframes"],
    )
    flash("Indicator removal started — parquet files will be reset to OHLCV only.", "info")
    return redirect(url_for("indicators.indicators_view", session_id=session_id))
