from flask import Blueprint, flash, redirect, render_template, request, url_for

from ..utils import db

bp = Blueprint("pairlist", __name__)


@bp.route("/<session_id>")
def pairlist_view(session_id):
    session = db.get_session(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    pairs = db.list_pairs(session_id)
    min_vol = request.args.get("min_vol", 0, type=float)
    search = request.args.get("search", "").upper()

    if min_vol > 0:
        pairs = [p for p in pairs if p["volume_24h"] >= min_vol]
    if search:
        pairs = [p for p in pairs if search in p["symbol"]]

    return render_template("pairlist/view.html",
                           session=session,
                           pairs=pairs,
                           min_vol=min_vol,
                           search=search)


@bp.route("/<session_id>/toggle", methods=["POST"])
def toggle_pair(session_id):
    symbol = request.form.get("symbol")
    excluded = request.form.get("excluded", "false").lower() == "true"
    db.set_pair_excluded(session_id, symbol, excluded)
    pairs = db.list_pairs(session_id)
    return render_template("partials/pair_row.html", pair=next(
        (p for p in pairs if p["symbol"] == symbol), None
    ), session_id=session_id)


@bp.route("/<session_id>/exclude_all_below", methods=["POST"])
def exclude_all_below(session_id):
    min_vol = float(request.form.get("min_vol", 0))
    pairs = db.list_pairs(session_id)
    for p in pairs:
        if p["volume_24h"] < min_vol:
            db.set_pair_excluded(session_id, p["symbol"], True)
    flash(f"Excluded pairs with volume < {min_vol:,.0f}.", "info")
    return redirect(url_for("pairlist.pairlist_view", session_id=session_id))


@bp.route("/<session_id>/reset_exclusions", methods=["POST"])
def reset_exclusions(session_id):
    pairs = db.list_pairs(session_id)
    for p in pairs:
        db.set_pair_excluded(session_id, p["symbol"], False)
    flash("All pair exclusions cleared.", "info")
    return redirect(url_for("pairlist.pairlist_view", session_id=session_id))
