from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.entry_logic_analyzer import run_entry_analysis
from ..utils import db

bp = Blueprint("entry_logic", __name__)


@bp.route("/<session_id>")
def entry_logic_view(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "path_a_analysis")

    return render_template("entry_logic/view.html",
                           session=session,
                           task=task)


@bp.route("/<session_id>/save", methods=["POST"])
def save_entry_logic(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    long_logic = request.form.get("entry_logic_long", "").strip()
    short_logic = request.form.get("entry_logic_short", "").strip()

    if not long_logic and not short_logic:
        flash("At least one entry condition (long or short) is required. Use Skip for Path B.", "error")
        return redirect(url_for("entry_logic.entry_logic_view", session_id=session_id))

    # Build combined entry_logic for backward compat with analyzer
    parts = []
    if long_logic:
        parts.append(f"# direction: long\n{long_logic}")
    if short_logic:
        parts.append(f"# direction: short\n{short_logic}")
    combined = "\n\n".join(parts)

    db.save_entry_logic(session_id, combined, "path_a",
                        entry_logic_long=long_logic,
                        entry_logic_short=short_logic)
    flash("Entry logic saved.", "success")
    return redirect(url_for("entry_logic.entry_logic_view", session_id=session_id))


@bp.route("/<session_id>/skip", methods=["POST"])
def skip_to_path_b(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    db.save_entry_logic(session_id, "", "path_b")
    flash("Skipped entry logic — using Path B (Optuna free search).", "info")
    return redirect(url_for("ic_analysis.ic_view", session_id=session_id))


@bp.route("/<session_id>/analyze", methods=["POST"])
def run_analysis(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    if session.get("entry_mode") != "path_a" or not session.get("entry_logic", "").strip():
        flash("No entry logic defined. Define entry logic first.", "error")
        return redirect(url_for("entry_logic.entry_logic_view", session_id=session_id))

    hold_bars = int(request.form.get("hold_bars", 10))
    min_profit_pct = float(request.form.get("min_profit_pct", 0.5))

    submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "path_a_analysis",
        run_entry_analysis,
        session_id,
        current_app.config["PARQUET_DIR"],
        session["timeframes"],
        session["entry_logic"],
        hold_bars,
        min_profit_pct,
    )

    flash("Entry logic analysis started.", "info")
    return redirect(url_for("entry_logic.entry_logic_view", session_id=session_id))


@bp.route("/<session_id>/stop", methods=["POST"])
def stop(session_id):
    from ..tasks.runner import request_cancel
    task = db.get_latest_task(session_id, "path_a_analysis")
    if task and task["status"] == "running":
        request_cancel(current_app.config["DB_PATH"], task["id"])
        flash("Stop requested.", "warning")
    return redirect(url_for("entry_logic.entry_logic_view", session_id=session_id))


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "path_a_analysis")
    return render_template("partials/task_progress.html", task=task)


@bp.route("/<session_id>/results")
def analysis_results(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "path_a_analysis")
    result = task.get("result", {}) if task else {}

    return render_template("entry_logic/results.html",
                           session=session,
                           task=task,
                           result=result)
