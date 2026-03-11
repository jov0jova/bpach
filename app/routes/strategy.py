from flask import Blueprint, current_app, flash, redirect, render_template, request, session as flask_session, url_for

from ..strategies.base import DEFAULT_STRATEGY_CODE
from ..utils import db

bp = Blueprint("strategy", __name__)

# Store strategy code per session in Flask session (ephemeral, server-side)
_STRATEGY_STORE: dict[str, str] = {}
_PARAMS_STORE: dict[str, dict] = {}


def get_strategy_code(session_id: str) -> str:
    return _STRATEGY_STORE.get(session_id, DEFAULT_STRATEGY_CODE)


def get_strategy_params(session_id: str) -> dict:
    return _PARAMS_STORE.get(session_id, {})


@bp.route("/<session_id>")
def strategy_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    code = get_strategy_code(session_id)
    params = get_strategy_params(session_id)

    return render_template("strategy/view.html",
                           session=session,
                           code=code,
                           params=params)


@bp.route("/<session_id>/save", methods=["POST"])
def save_strategy(session_id):
    code = request.form.get("strategy_code", DEFAULT_STRATEGY_CODE)
    initial_capital = float(request.form.get("initial_capital",
                                             current_app.config["DEFAULT_INITIAL_CAPITAL"]))
    fee_rate = float(request.form.get("fee_rate",
                                      current_app.config["DEFAULT_FEE_RATE"]))
    slippage = float(request.form.get("slippage",
                                      current_app.config["DEFAULT_SLIPPAGE"]))
    position_size = float(request.form.get("position_size",
                                           current_app.config["DEFAULT_POSITION_SIZE"]))

    # Validate code compiles
    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as e:
        flash(f"Syntax error in strategy code: {e}", "error")
        return redirect(url_for("strategy.strategy_view", session_id=session_id))

    _STRATEGY_STORE[session_id] = code
    _PARAMS_STORE[session_id] = {
        "initial_capital": initial_capital,
        "fee_rate": fee_rate,
        "slippage": slippage,
        "position_size": position_size,
    }

    flash("Strategy saved.", "success")
    return redirect(url_for("strategy.strategy_view", session_id=session_id))


@bp.route("/<session_id>/reset", methods=["POST"])
def reset_strategy(session_id):
    _STRATEGY_STORE.pop(session_id, None)
    _PARAMS_STORE.pop(session_id, None)
    flash("Strategy reset to default.", "info")
    return redirect(url_for("strategy.strategy_view", session_id=session_id))
