"""
Backtest routes — full-fledged standalone backtesting module.

Supports:
  • Custom entry/exit Python code
  • Dynamic pairlist selection
  • Fixed / ATR-based stop loss
  • Fixed / ATR-based / Risk:Reward take profit
  • Fixed / ATR-based trailing stop
  • Position sizing: fixed % | Kelly | ATR-risk
  • Walk-forward (rolling or anchored)
  • Saved configs
"""
from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..tasks.runner import submit_task
from ..services.backtest import run_backtest
from ..utils import db
from .. import models as m

bp = Blueprint("backtest", __name__)


def _parse_config_from_form(form, app_config) -> dict:
    def _f(key, default):
        v = form.get(key, "")
        try:
            return float(v) if str(v).strip() else default
        except (ValueError, AttributeError):
            return default

    def _i(key, default):
        v = form.get(key, "")
        try:
            return int(v) if str(v).strip() else default
        except (ValueError, AttributeError):
            return default

    return {
        "initial_capital":    _f("initial_capital", app_config["DEFAULT_INITIAL_CAPITAL"]),
        "fee_rate":            _f("fee_rate", app_config["DEFAULT_FEE_RATE"]),
        "slippage":            _f("slippage", app_config["DEFAULT_SLIPPAGE"]),
        "position_sizing":     form.get("position_sizing", "fixed"),
        "position_size":       _f("position_size", 10.0) / 100,
        "kelly_fraction":      _f("kelly_fraction", 25.0) / 100,
        "atr_risk_pct":        _f("atr_risk_pct", 1.0) / 100,
        "sl_mode":             form.get("sl_mode", "none"),
        "sl_pct":              _f("sl_pct", 2.0) / 100,
        "sl_atr_period":       _i("sl_atr_period", 14),
        "sl_atr_multiplier":   _f("sl_atr_multiplier", 2.0),
        "tp_mode":             form.get("tp_mode", "none"),
        "tp_pct":              _f("tp_pct", 4.0) / 100,
        "tp_atr_period":       _i("tp_atr_period", 14),
        "tp_atr_multiplier":   _f("tp_atr_multiplier", 4.0),
        "tp_rr_ratio":         _f("tp_rr_ratio", 2.0),
        "trail_mode":          form.get("trail_mode", "none"),
        "trail_pct":           _f("trail_pct", 2.0) / 100,
        "trail_atr_period":    _i("trail_atr_period", 14),
        "trail_atr_multiplier":_f("trail_atr_multiplier", 1.5),
        "wfo_splits":          _i("wfo_splits", 5),
        "wfo_train_ratio":     _f("wfo_train_ratio", 70.0) / 100,
        "wfo_anchored":        form.get("wfo_anchored") == "1",
    }


@bp.route("/<session_id>")
def backtest_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    task         = db.get_latest_task(session_id, "backtest")
    runs         = db.list_backtest_runs(session_id)
    pairs        = m.list_pairs(current_app.config["DB_PATH"], session_id)
    active_pairs = [p for p in pairs if not p["excluded"] and p["candle_count"] > 0]
    saved_cfg    = m.get_latest_backtest_config(current_app.config["DB_PATH"], session_id)

    return render_template("backtest/view.html",
                           session=session, task=task, runs=runs,
                           active_pairs=active_pairs, saved_cfg=saved_cfg)


@bp.route("/<session_id>/run", methods=["POST"])
def run(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    entry_code     = request.form.get("entry_code", "").strip()
    exit_code      = request.form.get("exit_code", "").strip()
    selected_pairs = request.form.getlist("selected_pairs")
    config         = _parse_config_from_form(request.form, current_app.config)

    # Save config for next session
    cfg_data = {
        "session_id":     session_id,
        "name":           request.form.get("config_name", "Default"),
        "entry_code":     entry_code,
        "exit_code":      exit_code,
        "selected_pairs": selected_pairs,
        **config,
    }
    m.upsert_backtest_config(current_app.config["DB_PATH"], session_id, cfg_data)

    strategy_code = ""
    if entry_code:
        strategy_code += entry_code + "\n"
    if exit_code:
        strategy_code += exit_code + "\n"

    params = {}
    run_id = db.create_backtest_run(session_id, strategy_code, params)

    submit_task(
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
        selected_pairs if selected_pairs else None,
    )

    flash("Backtest started.", "info")
    return redirect(url_for("backtest.backtest_view", session_id=session_id))


@bp.route("/<session_id>/result/<run_id>")
def result(session_id, run_id):
    from ..utils.plotting import pnl_distribution

    session = db.get_session(session_id)
    run     = db.get_backtest_run(run_id)
    if not run:
        flash("Backtest run not found.", "error")
        return redirect(url_for("backtest.backtest_view", session_id=session_id))

    trades  = db.get_trades(run_id)
    winners = [t for t in trades if t["is_winner"]]
    losers  = [t for t in trades if not t["is_winner"]]

    pnl_values = [t["pnl_pct"] for t in trades]
    pnl_chart  = pnl_distribution(pnl_values, "PnL Distribution") if pnl_values else None

    task = db.get_latest_task(session_id, "backtest")
    task_result = task.get("result", {}) if task else {}

    return render_template("backtest/result.html",
                           session=session, run=run,
                           trades=trades[:500],
                           winners=winners, losers=losers,
                           pnl_chart=pnl_chart,
                           task_result=task_result)


@bp.route("/<session_id>/stop", methods=["POST"])
def stop(session_id):
    from ..tasks.runner import request_cancel
    task = db.get_latest_task(session_id, "backtest")
    if task and task["status"] == "running":
        request_cancel(current_app.config["DB_PATH"], task["id"])
        flash("Stop requested.", "warning")
    return redirect(url_for("backtest.backtest_view", session_id=session_id))


@bp.route("/<session_id>/status")
def status(session_id):
    task = db.get_latest_task(session_id, "backtest")
    return render_template("partials/task_progress.html", task=task)
