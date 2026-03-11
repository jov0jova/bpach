from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.algofinder import run_algofinder
from ..utils import db

bp = Blueprint("algofinder", __name__)


@bp.route("/<session_id>")
def algofinder_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "algofinder")
    results = db.list_algo_results(session_id)

    return render_template("algofinder/view.html",
                           session=session,
                           task=task,
                           results=results)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    n_trials = int(request.form.get("n_trials", current_app.config["OPTUNA_TRIALS"]))
    config = {
        "initial_capital": float(request.form.get("initial_capital",
                                                    current_app.config["DEFAULT_INITIAL_CAPITAL"])),
        "fee_rate": current_app.config["DEFAULT_FEE_RATE"],
        "slippage": current_app.config["DEFAULT_SLIPPAGE"],
        "position_size": current_app.config["DEFAULT_POSITION_SIZE"],
        "wfo_splits": 3,
        "wfo_train_ratio": 0.7,
    }

    task_id = submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "algofinder",
        run_algofinder,
        session_id,
        current_app.config["PARQUET_DIR"],
        session["timeframes"],
        n_trials,
        config,
    )

    flash(f"Algo Finder started with {n_trials} trials.", "info")
    return redirect(url_for("algofinder.algofinder_view", session_id=session_id))


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "algofinder")
    return render_template("partials/task_progress.html", task=task)
