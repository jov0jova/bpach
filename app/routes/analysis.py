import json

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.analysis import (
    run_analysis,
    compute_indicator_stats,
    compute_win_rate_by_condition,
    compute_correlation_matrix,
)
from ..utils import db
from ..utils.plotting import (
    candlestick_chart,
    indicator_heatmap,
    win_rate_bar_chart,
)
from ..utils.parquet import parquet_path
import pandas as pd

bp = Blueprint("analysis", __name__)


@bp.route("/<session_id>")
def analysis_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    runs = db.list_backtest_runs(session_id)
    task = db.get_latest_task(session_id, "analysis")
    selected_run_id = request.args.get("run_id")

    return render_template("analysis/view.html",
                           session=session,
                           runs=runs,
                           task=task,
                           selected_run_id=selected_run_id)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    run_id = request.form.get("run_id")
    if not run_id:
        runs = db.list_backtest_runs(session_id)
        done = [r for r in runs if r["status"] == "done"]
        if not done:
            flash("No completed backtest found. Run a backtest first.", "warning")
            return redirect(url_for("analysis.analysis_view", session_id=session_id))
        run_id = done[0]["id"]

    task_id = submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "analysis",
        run_analysis,
        session_id,
        current_app.config["PARQUET_DIR"],
        run_id,
        session["timeframes"],
    )

    flash("Analysis started.", "info")
    return redirect(url_for("analysis.analysis_view",
                            session_id=session_id, run_id=run_id))


@bp.route("/<session_id>/results/<run_id>")
def results(session_id, run_id):
    session = db.get_session(session_id)
    run = db.get_backtest_run(run_id)
    if not run:
        flash("Backtest run not found.", "error")
        return redirect(url_for("analysis.analysis_view", session_id=session_id))

    analyses = db.get_entry_analyses(run_id)
    primary_tf = session["timeframes"][0] if session["timeframes"] else "1h"

    # Compute stats
    ind_stats = compute_indicator_stats(analyses, primary_tf)
    win_rate_conditions = compute_win_rate_by_condition(analyses, [], primary_tf)
    corr_matrix = compute_correlation_matrix(analyses, primary_tf)

    # Charts
    heatmap_chart = indicator_heatmap(corr_matrix, "Indicator Correlation at Winners") if corr_matrix else None
    wr_chart = win_rate_bar_chart(win_rate_conditions, "Condition Frequency in Winners") if win_rate_conditions else None

    # Key indicator stats for display (top 10 most variable)
    key_stats = sorted(ind_stats.items(), key=lambda x: x[1]["std"], reverse=True)[:15]

    return render_template("analysis/results.html",
                           session=session,
                           run=run,
                           analyses=analyses[:50],
                           ind_stats=ind_stats,
                           key_stats=key_stats,
                           win_rate_conditions=win_rate_conditions,
                           heatmap_chart=heatmap_chart,
                           wr_chart=wr_chart,
                           total_winners=len(analyses),
                           primary_tf=primary_tf)


@bp.route("/<session_id>/trade/<trade_id>")
def trade_detail(session_id, trade_id):
    """Show candlestick chart for a specific winning trade."""
    session = db.get_session(session_id)
    if not session:
        return "Session not found", 404

    analyses = db.get_entry_analyses(request.args.get("run_id", ""))
    analysis = next((a for a in analyses if a["trade_id"] == trade_id), None)
    if not analysis:
        return "Analysis not found", 404

    symbol = analysis["symbol"]
    primary_tf = session["timeframes"][0] if session["timeframes"] else "1h"

    path = parquet_path(current_app.config["PARQUET_DIR"], session_id, symbol, primary_tf)
    chart = None
    if path.exists():
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Show ±100 candles around entry
        entry_time = pd.Timestamp(analysis["entry_time"])
        entry_idx = (df["timestamp"] - entry_time).abs().idxmin()
        start = max(0, entry_idx - 100)
        end = min(len(df), entry_idx + 50)
        chart_df = df.iloc[start:end]
        chart = candlestick_chart(chart_df, symbol, primary_tf,
                                  entry_times=[entry_time])

    return render_template("analysis/trade_detail.html",
                           session=session,
                           analysis=analysis,
                           chart=chart,
                           symbol=symbol)


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "analysis")
    return render_template("partials/task_progress.html", task=task)
