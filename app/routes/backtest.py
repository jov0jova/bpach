import json

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.backtest import run_backtest
from ..routes.strategy import get_strategy_code, get_strategy_params
from ..utils import db
from ..utils.plotting import equity_curve_chart, pnl_distribution

bp = Blueprint("backtest", __name__)


@bp.route("/<session_id>")
def backtest_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task = db.get_latest_task(session_id, "backtest")
    runs = db.list_backtest_runs(session_id)

    return render_template("backtest/view.html",
                           session=session,
                           task=task,
                           runs=runs)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    strategy_code = get_strategy_code(session_id)
    params = get_strategy_params(session_id)
    config = {
        "initial_capital": params.get("initial_capital", current_app.config["DEFAULT_INITIAL_CAPITAL"]),
        "fee_rate": params.get("fee_rate", current_app.config["DEFAULT_FEE_RATE"]),
        "slippage": params.get("slippage", current_app.config["DEFAULT_SLIPPAGE"]),
        "position_size": params.get("position_size", current_app.config["DEFAULT_POSITION_SIZE"]),
        "wfo_splits": current_app.config["WFO_SPLITS"],
        "wfo_train_ratio": current_app.config["WFO_TRAIN_RATIO"],
    }

    run_id = db.create_backtest_run(session_id, strategy_code, params)

    task_id = submit_task(
        current_app.config["DB_PATH"],
        session_id,
        "backtest",
        run_backtest,
        session_id,
        current_app.config["PARQUET_DIR"],
        run_id,
        strategy_code,
        params,
        session["timeframes"],
        config,
    )

    flash("Backtest started.", "info")
    return redirect(url_for("backtest.backtest_view", session_id=session_id))


@bp.route("/<session_id>/result/<run_id>")
def result(session_id, run_id):
    session = db.get_session(session_id)
    run = db.get_backtest_run(run_id)
    if not run:
        flash("Backtest run not found.", "error")
        return redirect(url_for("backtest.backtest_view", session_id=session_id))

    trades = db.get_trades(run_id)
    winners = [t for t in trades if t["is_winner"]]
    losers = [t for t in trades if not t["is_winner"]]

    pnl_values = [t["pnl_pct"] for t in trades]
    pnl_chart = pnl_distribution(pnl_values, "PnL Distribution") if pnl_values else None

    return render_template("backtest/result.html",
                           session=session,
                           run=run,
                           trades=trades[:200],  # limit for display
                           winners=winners,
                           losers=losers,
                           pnl_chart=pnl_chart)


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "backtest")
    return render_template("partials/task_progress.html", task=task)
