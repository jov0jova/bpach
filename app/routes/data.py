from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.download import run_download
from ..utils import db

bp = Blueprint("data", __name__)


@bp.route("/<session_id>")
def data_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "download")
    pairs = db.list_pairs(session_id)
    return render_template("data/view.html",
                           session=session,
                           task=task,
                           pairs=pairs,
                           pair_count=len(pairs))


@bp.route("/<session_id>/download", methods=["POST"])
def start_download(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    lookback_days = int(request.form.get("lookback_days", 365))
    max_pairs = int(request.form.get("max_pairs", current_app.config["MAX_PAIRS"]))

    task_id = submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "download",
        run_download,
        session_id,
        current_app.config["PARQUET_DIR"],
        session["exchange"],
        session["timeframes"],
        session["quote_asset"],
        max_pairs,
        lookback_days,
    )

    flash("Data download started.", "info")
    return redirect(url_for("data.data_view", session_id=session_id))


@bp.route("/<session_id>/stop", methods=["POST"])
def stop(session_id):
    from ..tasks.runner import request_cancel
    task = db.get_latest_task(session_id, "download")
    if task and task["status"] == "running":
        request_cancel(current_app.config["DB_PATH"], task["id"])
        flash("Stop requested.", "warning")
    return redirect(url_for("data.data_view", session_id=session_id))


@bp.route("/<session_id>/status")
def download_status(session_id):
    """HTMX polling for download progress."""
    task = db.get_latest_task(session_id, "download")
    return render_template("partials/task_progress.html", task=task)
