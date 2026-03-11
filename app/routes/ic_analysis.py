from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.ic_analysis import run_ic_analysis
from ..utils import db

bp = Blueprint("ic_analysis", __name__)


@bp.route("/<session_id>")
def ic_view(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "ic_analysis")
    result = task.get("result", {}) if task and task.get("status") == "done" else {}

    return render_template("ic_analysis/view.html",
                           session=session,
                           task=task,
                           result=result)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "ic_analysis",
        run_ic_analysis,
        session_id,
        current_app.config["PARQUET_DIR"],
        session["timeframes"],
    )

    flash("IC Analysis started. This may take a few minutes.", "info")
    return redirect(url_for("ic_analysis.ic_view", session_id=session_id))


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "ic_analysis")
    return render_template("partials/task_progress.html", task=task)
