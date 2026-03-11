import json
from datetime import datetime, timezone

from flask import Blueprint, flash, redirect, render_template, request, url_for

from ..utils import db

bp = Blueprint("pairlist", __name__)


@bp.route("/<session_id>")
def pairlist_view(session_id):
    session = db.get_session_extended(session_id)
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


@bp.route("/<session_id>/save_config", methods=["POST"])
def save_config(session_id):
    """Save and apply a dynamic pairlist config (VolumePairList-style JSON)."""
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    config_json = request.form.get("pairlist_config", "[]").strip()
    try:
        config = json.loads(config_json)
        if not isinstance(config, list):
            raise ValueError("Config must be a JSON array.")
    except (json.JSONDecodeError, ValueError) as e:
        flash(f"Invalid JSON: {e}", "error")
        return redirect(url_for("pairlist.pairlist_view", session_id=session_id))

    db.save_pairlist_config(session_id, config)
    flash("Pairlist config saved.", "success")
    return redirect(url_for("pairlist.apply_config", session_id=session_id))


@bp.route("/<session_id>/apply_config", methods=["GET", "POST"])
def apply_config(session_id):
    """Apply the saved dynamic pairlist config to exclude/include pairs."""
    session = db.get_session_extended(session_id)
    if not session:
        flash("Session not found.", "error")
        return redirect(url_for("sessions.list_sessions"))

    config = session.get("pairlist_config", [])
    if not config:
        flash("No pairlist config saved. Paste your config first.", "warning")
        return redirect(url_for("pairlist.pairlist_view", session_id=session_id))

    pairs = db.list_pairs(session_id)

    # Parse filters from config
    number_assets = 80
    min_volume = 0.0
    offset = 0
    final_count = None
    min_days = None

    for step in config:
        method = step.get("method", "")
        if method == "VolumePairList":
            number_assets = step.get("number_assets", 80)
            min_volume = step.get("min_value", 0.0)
        elif method == "AgeFilter":
            min_days = step.get("min_days_listed", None)
        elif method == "OffsetFilter":
            offset = step.get("offset", 0)
            if "number_assets" in step:
                final_count = step["number_assets"]

    # Sort by volume descending, apply VolumePairList
    sorted_pairs = sorted(pairs, key=lambda p: p["volume_24h"], reverse=True)

    # Apply min volume
    if min_volume > 0:
        sorted_pairs = [p for p in sorted_pairs if p["volume_24h"] >= min_volume]

    # Apply number_assets cap
    sorted_pairs = sorted_pairs[:number_assets]

    # Apply AgeFilter (require data_start to be old enough)
    if min_days is not None:
        now = datetime.now(timezone.utc)
        filtered = []
        for p in sorted_pairs:
            if p["data_start"] is None:
                filtered.append(p)  # no data yet, keep it
                continue
            ds = p["data_start"]
            if hasattr(ds, "replace"):
                ds = ds.replace(tzinfo=timezone.utc) if ds.tzinfo is None else ds
            age_days = (now - ds).days if hasattr(ds, "days") else 9999
            if age_days >= min_days:
                filtered.append(p)
        sorted_pairs = filtered

    # Apply OffsetFilter
    if offset:
        sorted_pairs = sorted_pairs[offset:]
    if final_count:
        sorted_pairs = sorted_pairs[:final_count]

    active_symbols = {p["symbol"] for p in sorted_pairs}

    # Apply: exclude pairs NOT in active set
    excluded_count = 0
    included_count = 0
    for p in pairs:
        should_exclude = p["symbol"] not in active_symbols
        db.set_pair_excluded(session_id, p["symbol"], should_exclude)
        if should_exclude:
            excluded_count += 1
        else:
            included_count += 1

    flash(f"Dynamic config applied: {included_count} pairs active, {excluded_count} excluded.", "success")
    return redirect(url_for("pairlist.pairlist_view", session_id=session_id))
