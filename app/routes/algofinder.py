from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.algofinder import (
    INDICATOR_CATALOG, DEFAULT_INDICATORS, STRATEGY_TEMPLATES,
    build_dynamic_catalog, run_algofinder, run_algofinder_path_a,
)
from ..utils import db
from .. import models as m

bp = Blueprint("algofinder", __name__)


@bp.route("/<session_id>")
def algofinder_view(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task    = db.get_latest_task(session_id, "algofinder")
    results = db.list_algo_results(session_id)

    path_a_task   = db.get_latest_task(session_id, "path_a_analysis")
    path_a_result = (path_a_task.get("result", {})
                     if path_a_task and path_a_task.get("status") == "done" else {})

    # IC analysis results for IC-guided mode
    ic_task   = db.get_latest_task(session_id, "ic_analysis")
    ic_result = (ic_task.get("result", {})
                 if ic_task and ic_task.get("status") == "done" else {})

    session_tfs     = session.get("timeframes") or ["1h"]
    session_signal_tf = session_tfs[0]
    session_htf_tfs   = session_tfs[1:]

    primary_tf = session_signal_tf
    catalog    = build_dynamic_catalog(
        current_app.config["PARQUET_DIR"], session_id, primary_tf)
    default_inds = [k for k, v in catalog.items() if v.get("default")]
    if not default_inds:
        default_inds = DEFAULT_INDICATORS

    all_pairs    = m.list_pairs(current_app.config["DB_PATH"], session_id)
    active_pairs = [p for p in all_pairs if not p["excluded"] and p["candle_count"] > 0]

    all_timeframes = ["1m","3m","5m","15m","30m","1h","2h","4h","8h","12h","1d","3d","1w"]

    return render_template("algofinder/view.html",
                           session=session, task=task, results=results,
                           path_a_result=path_a_result, ic_result=ic_result,
                           indicator_catalog=catalog,
                           default_indicators=default_inds,
                           active_pairs=active_pairs,
                           strategy_templates=STRATEGY_TEMPLATES,
                           all_timeframes=all_timeframes,
                           session_signal_tf=session_signal_tf,
                           session_htf_tfs=session_htf_tfs)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    n_trials       = int(request.form.get("n_trials", current_app.config["OPTUNA_TRIALS"]))
    mode           = request.form.get("mode", session.get("entry_mode", "path_b"))
    template       = request.form.get("template", "free")
    multi_objective = request.form.get("multi_objective") == "1"

    selected_indicators = request.form.getlist("indicators") or DEFAULT_INDICATORS
    selected_pairs      = request.form.getlist("pairs") or []

    # Timeframe override — form values take priority over session defaults
    _session_tfs  = session.get("timeframes") or ["1h"]
    signal_tf     = request.form.get("signal_tf") or _session_tfs[0]
    htf_list      = request.form.getlist("htf_tfs") or _session_tfs[1:]
    # Deduplicate and ensure signal TF is not in HTF list
    run_timeframes = [signal_tf] + [tf for tf in htf_list if tf != signal_tf]

    # Execution TF — faster TF for realistic entry/exit pricing (e.g. 1m)
    execution_tf = request.form.get("execution_tf") or None
    if execution_tf == signal_tf or execution_tf == "":
        execution_tf = None

    # IC analysis results — passed to algofinder for guided sampling
    ic_task   = db.get_latest_task(session_id, "ic_analysis")
    ic_result = (ic_task.get("result", {})
                 if ic_task and ic_task.get("status") == "done" else {})

    config = {
        "initial_capital":    float(request.form.get("initial_capital",
                                                      current_app.config["DEFAULT_INITIAL_CAPITAL"])),
        "fee_rate":           current_app.config["DEFAULT_FEE_RATE"],
        "slippage":           current_app.config["DEFAULT_SLIPPAGE"],
        "position_size":      current_app.config["DEFAULT_POSITION_SIZE"],
        "wfo_splits":         3,
        "wfo_train_ratio":    0.7,
        "selected_indicators": selected_indicators,
        "selected_pairs":      selected_pairs,
        "template":            template,
        "multi_objective":     multi_objective,
        "ic_result":           ic_result,
        "min_oos_trades":      int(request.form.get("min_oos_trades", 5)),
        "n_jobs":              int(request.form.get("n_jobs", 2)),
        "max_pairs":           int(request.form.get("max_pairs", 30)),
        "execution_tf":        execution_tf,
    }

    if mode == "path_a" and session.get("entry_logic"):
        path_a_task = db.get_latest_task(session_id, "path_a_analysis")
        if not path_a_task or path_a_task.get("status") != "done":
            flash("Run Entry Logic Analysis first (Step 4) before using Path A.", "error")
            return redirect(url_for("algofinder.algofinder_view", session_id=session_id))

        analysis_result = path_a_task.get("result", {})
        if not analysis_result.get("top_indicators"):
            flash("No discriminative indicators found. Re-run the entry analysis.", "error")
            return redirect(url_for("algofinder.algofinder_view", session_id=session_id))

        submit_task(
            current_app.config["DB_PATH"], session_id, "algofinder",
            run_algofinder_path_a,
            session_id, current_app.config["PARQUET_DIR"],
            run_timeframes, session["entry_logic"],
            analysis_result, n_trials, config,
        )
        flash(f"Path A Algo Finder started — {n_trials} trials tuning your entry filters.", "info")
    else:
        ic_msg = " (IC-guided)" if ic_result else ""
        multi_msg = " (multi-objective)" if multi_objective else ""
        submit_task(
            current_app.config["DB_PATH"], session_id, "algofinder",
            run_algofinder,
            session_id, current_app.config["PARQUET_DIR"],
            run_timeframes, n_trials, config,
        )
        htf_str = " + HTF: " + ", ".join(run_timeframes[1:]) if len(run_timeframes) > 1 else ""
        flash(
            f"Algo Finder started — {n_trials} trials | Signal TF: {signal_tf}{htf_str} | "
            f"Template: {STRATEGY_TEMPLATES.get(template, {}).get('label', template)}"
            f"{ic_msg}{multi_msg}",
            "info",
        )

    return redirect(url_for("algofinder.algofinder_view", session_id=session_id))


@bp.route("/<session_id>/clear", methods=["POST"])
def clear(session_id):
    m.clear_algo_results(current_app.config["DB_PATH"], session_id)
    flash("Results cleared.", "info")
    return redirect(url_for("algofinder.algofinder_view", session_id=session_id))


@bp.route("/<session_id>/stop", methods=["POST"])
def stop(session_id):
    from ..tasks.runner import request_cancel
    task = db.get_latest_task(session_id, "algofinder")
    if task and task["status"] == "running":
        request_cancel(current_app.config["DB_PATH"], task["id"])
        flash("Stop requested.", "warning")
    return redirect(url_for("algofinder.algofinder_view", session_id=session_id))


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "algofinder")
    return render_template("partials/task_progress.html", task=task)
