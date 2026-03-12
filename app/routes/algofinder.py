from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.algofinder import (
    INDICATOR_CATALOG, DEFAULT_INDICATORS,
    run_algofinder, run_algofinder_path_a,
)
from ..utils import db

bp = Blueprint("algofinder", __name__)


@bp.route("/<session_id>")
def algofinder_view(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "algofinder")
    results = db.list_algo_results(session_id)

    path_a_task = db.get_latest_task(session_id, "path_a_analysis")
    path_a_result = (path_a_task.get("result", {})
                     if path_a_task and path_a_task.get("status") == "done" else {})

    return render_template("algofinder/view.html",
                           session=session,
                           task=task,
                           results=results,
                           path_a_result=path_a_result,
                           indicator_catalog=INDICATOR_CATALOG,
                           default_indicators=DEFAULT_INDICATORS)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    n_trials = int(request.form.get("n_trials", current_app.config["OPTUNA_TRIALS"]))
    mode = request.form.get("mode", session.get("entry_mode", "path_b"))

    # Indicator selection (Path B only; checkboxes send a list of keys)
    selected_indicators = request.form.getlist("indicators") or DEFAULT_INDICATORS

    config = {
        "initial_capital": float(request.form.get("initial_capital",
                                                    current_app.config["DEFAULT_INITIAL_CAPITAL"])),
        "fee_rate": current_app.config["DEFAULT_FEE_RATE"],
        "slippage": current_app.config["DEFAULT_SLIPPAGE"],
        "position_size": current_app.config["DEFAULT_POSITION_SIZE"],
        "wfo_splits": 3,
        "wfo_train_ratio": 0.7,
        "selected_indicators": selected_indicators,
    }

    if mode == "path_a" and session.get("entry_logic"):
        path_a_task = db.get_latest_task(session_id, "path_a_analysis")
        if not path_a_task or path_a_task.get("status") != "done":
            flash("Run Entry Logic Analysis first (Phase 4) before using Path A.", "error")
            return redirect(url_for("algofinder.algofinder_view", session_id=session_id))

        analysis_result = path_a_task.get("result", {})
        if not analysis_result.get("top_indicators"):
            flash("No discriminative indicators found. Re-run the entry analysis.", "error")
            return redirect(url_for("algofinder.algofinder_view", session_id=session_id))

        submit_task(
            current_app.config["DB_PATH"],
            session_id,
            "algofinder",
            run_algofinder_path_a,
            session_id,
            current_app.config["PARQUET_DIR"],
            session["timeframes"],
            session["entry_logic"],
            analysis_result,
            n_trials,
            config,
        )
        flash(f"Path A Algo Finder started — {n_trials} trials tuning your entry filters.", "info")
    else:
        submit_task(
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
        flash(f"Path B Algo Finder started — {n_trials} trials free search.", "info")

    return redirect(url_for("algofinder.algofinder_view", session_id=session_id))


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "algofinder")
    return render_template("partials/task_progress.html", task=task)
