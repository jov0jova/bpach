"""
DuckDB schema definitions and CRUD helpers.
All tables are created in data/app.db on first run.
"""
import json
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb

# DuckDB does not support concurrent writers from multiple threads.
# Serialise every DB access behind a single lock so progress callbacks
# (Optuna, indicator loops, etc.) and Flask request threads never conflict.
_db_lock = threading.Lock()


class _LockedConn:
    """Wraps a DuckDB connection and holds _db_lock for its lifetime.

    Existing call sites use the pattern:
        con = _conn(db_path)
        con.execute(...)
        con.close()
    This wrapper acquires the lock on construction and releases it in close(),
    so no call sites need to change.
    """
    def __init__(self, db_path: str | Path):
        _db_lock.acquire()
        self._con = duckdb.connect(str(db_path))

    def execute(self, *args, **kwargs):
        return self._con.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._con.executemany(*args, **kwargs)

    def close(self):
        try:
            self._con.close()
        finally:
            _db_lock.release()


def _conn(db_path: str | Path) -> _LockedConn:
    return _LockedConn(db_path)


@contextmanager
def _db(db_path: str | Path):
    """Context manager that opens a DuckDB connection and closes it on exit."""
    con = _conn(db_path)
    try:
        yield con
    finally:
        con.close()


def init_db(db_path: str | Path) -> None:
    con = _conn(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            exchange          TEXT NOT NULL DEFAULT 'binance',
            timeframes        TEXT NOT NULL DEFAULT '["15m","1h","4h"]',
            quote_asset       TEXT NOT NULL DEFAULT 'USDT',
            status            TEXT NOT NULL DEFAULT 'created',
            created_at        TIMESTAMP NOT NULL,
            updated_at        TIMESTAMP NOT NULL,
            notes             TEXT DEFAULT '',
            entry_logic       TEXT DEFAULT '',
            entry_mode        TEXT NOT NULL DEFAULT 'path_b',
            pairlist_config   TEXT DEFAULT '[]',
            entry_logic_long  TEXT DEFAULT '',
            entry_logic_short TEXT DEFAULT ''
        )
    """)
    # Migrate existing sessions tables that may be missing the new columns
    for col, typedef in [
        ("entry_logic",       "TEXT DEFAULT ''"),
        ("entry_mode",        "TEXT NOT NULL DEFAULT 'path_b'"),
        ("pairlist_config",   "TEXT DEFAULT '[]'"),
        ("entry_logic_long",  "TEXT DEFAULT ''"),
        ("entry_logic_short", "TEXT DEFAULT ''"),
    ]:
        try:
            con.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # column already exists

    con.execute("""
        CREATE TABLE IF NOT EXISTS pairs (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            base_asset      TEXT NOT NULL,
            quote_asset     TEXT NOT NULL,
            active          BOOLEAN NOT NULL DEFAULT TRUE,
            volume_24h      DOUBLE DEFAULT 0,
            excluded        BOOLEAN NOT NULL DEFAULT FALSE,
            data_start      TIMESTAMP,
            data_end        TIMESTAMP,
            candle_count    INTEGER DEFAULT 0,
            created_at      TIMESTAMP NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id                  TEXT PRIMARY KEY,
            session_id          TEXT NOT NULL,
            strategy_code       TEXT NOT NULL,
            strategy_params     TEXT NOT NULL DEFAULT '{}',
            status              TEXT NOT NULL DEFAULT 'pending',
            total_trades        INTEGER DEFAULT 0,
            win_rate            DOUBLE DEFAULT 0,
            profit_factor       DOUBLE DEFAULT 0,
            sharpe_ratio        DOUBLE DEFAULT 0,
            max_drawdown        DOUBLE DEFAULT 0,
            total_return        DOUBLE DEFAULT 0,
            avg_trade_duration  DOUBLE DEFAULT 0,
            oos_return          DOUBLE DEFAULT 0,
            oos_win_rate        DOUBLE DEFAULT 0,
            created_at          TIMESTAMP NOT NULL,
            completed_at        TIMESTAMP,
            error_msg           TEXT DEFAULT ''
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            session_id      TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            timeframe       TEXT NOT NULL,
            entry_time      TIMESTAMP NOT NULL,
            exit_time       TIMESTAMP,
            entry_price     DOUBLE NOT NULL,
            exit_price      DOUBLE,
            direction       TEXT NOT NULL DEFAULT 'long',
            pnl_pct         DOUBLE DEFAULT 0,
            pnl_abs         DOUBLE DEFAULT 0,
            duration_bars   INTEGER DEFAULT 0,
            is_winner       BOOLEAN DEFAULT FALSE,
            entry_signals   TEXT DEFAULT '{}',
            exit_reason     TEXT DEFAULT ''
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS entry_analysis (
            id              TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            trade_id        TEXT NOT NULL,
            session_id      TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            timeframe       TEXT NOT NULL,
            entry_time      TIMESTAMP NOT NULL,
            indicator_snapshot  TEXT NOT NULL DEFAULT '{}',
            candle_context  TEXT NOT NULL DEFAULT '{}',
            pattern_flags   TEXT NOT NULL DEFAULT '{}',
            created_at      TIMESTAMP NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            task_type   TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            progress    INTEGER NOT NULL DEFAULT 0,
            total       INTEGER NOT NULL DEFAULT 0,
            message     TEXT DEFAULT '',
            result      TEXT DEFAULT '{}',
            error       TEXT DEFAULT '',
            created_at  TIMESTAMP NOT NULL,
            updated_at  TIMESTAMP NOT NULL
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS algo_results (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            run_id          TEXT,
            rank            INTEGER NOT NULL,
            strategy_name   TEXT NOT NULL,
            params          TEXT NOT NULL DEFAULT '{}',
            rules_description TEXT DEFAULT '',
            is_return       DOUBLE DEFAULT 0,
            oos_return      DOUBLE DEFAULT 0,
            win_rate        DOUBLE DEFAULT 0,
            sharpe          DOUBLE DEFAULT 0,
            max_drawdown    DOUBLE DEFAULT 0,
            profit_factor   DOUBLE DEFAULT 0,
            calmar_ratio    DOUBLE DEFAULT 0,
            sortino_ratio   DOUBLE DEFAULT 0,
            expectancy      DOUBLE DEFAULT 0,
            pair_coverage   DOUBLE DEFAULT 0,
            regime          TEXT DEFAULT 'all',
            template        TEXT DEFAULT 'free',
            created_at      TIMESTAMP NOT NULL
        )
    """)
    # Migrate older algo_results tables missing new columns
    for col, typedef in [
        ("profit_factor",  "DOUBLE DEFAULT 0"),
        ("calmar_ratio",   "DOUBLE DEFAULT 0"),
        ("sortino_ratio",  "DOUBLE DEFAULT 0"),
        ("expectancy",     "DOUBLE DEFAULT 0"),
        ("pair_coverage",  "DOUBLE DEFAULT 0"),
        ("regime",         "TEXT DEFAULT 'all'"),
        ("template",       "TEXT DEFAULT 'free'"),
    ]:
        try:
            con.execute(f"ALTER TABLE algo_results ADD COLUMN {col} {typedef}")
        except Exception:
            pass

    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_configs (
            id                      TEXT PRIMARY KEY,
            session_id              TEXT NOT NULL,
            name                    TEXT NOT NULL DEFAULT 'Default',
            entry_code              TEXT DEFAULT '',
            exit_code               TEXT DEFAULT '',
            selected_pairs          TEXT DEFAULT '[]',
            initial_capital         DOUBLE DEFAULT 10000,
            fee_rate                DOUBLE DEFAULT 0.001,
            slippage                DOUBLE DEFAULT 0.0005,
            position_sizing         TEXT DEFAULT 'fixed',
            position_size           DOUBLE DEFAULT 0.1,
            kelly_fraction          DOUBLE DEFAULT 0.25,
            atr_risk_pct            DOUBLE DEFAULT 1.0,
            sl_mode                 TEXT DEFAULT 'none',
            sl_pct                  DOUBLE DEFAULT 2.0,
            sl_atr_period           INTEGER DEFAULT 14,
            sl_atr_multiplier       DOUBLE DEFAULT 2.0,
            tp_mode                 TEXT DEFAULT 'none',
            tp_pct                  DOUBLE DEFAULT 4.0,
            tp_atr_period           INTEGER DEFAULT 14,
            tp_atr_multiplier       DOUBLE DEFAULT 4.0,
            tp_rr_ratio             DOUBLE DEFAULT 2.0,
            trail_mode              TEXT DEFAULT 'none',
            trail_pct               DOUBLE DEFAULT 2.0,
            trail_atr_period        INTEGER DEFAULT 14,
            trail_atr_multiplier    DOUBLE DEFAULT 1.5,
            wfo_splits              INTEGER DEFAULT 5,
            wfo_train_ratio         DOUBLE DEFAULT 0.7,
            created_at              TIMESTAMP NOT NULL,
            updated_at              TIMESTAMP NOT NULL
        )
    """)

    # Extend backtest_runs with richer metrics
    for col, typedef in [
        ("profit_factor_oos", "DOUBLE DEFAULT 0"),
        ("calmar_ratio",      "DOUBLE DEFAULT 0"),
        ("sortino_ratio",     "DOUBLE DEFAULT 0"),
        ("expectancy",        "DOUBLE DEFAULT 0"),
        ("recovery_factor",   "DOUBLE DEFAULT 0"),
        ("config_id",         "TEXT DEFAULT ''"),
        ("config_snapshot",   "TEXT DEFAULT '{}'"),
        ("pairs_backtested",  "INTEGER DEFAULT 0"),
        ("pairs_profitable",  "INTEGER DEFAULT 0"),
    ]:
        try:
            con.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {typedef}")
        except Exception:
            pass

    con.close()


# ── Sessions ────────────────────────────────────────────────────────────────

def create_session(db_path, name: str, exchange: str, timeframes: list, quote_asset: str) -> dict:
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    con = _conn(db_path)
    con.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [sid, name, exchange, json.dumps(timeframes), quote_asset, "created", now, now, "", "", "path_b", "[]", "", ""]
    )
    con.close()
    return get_session_extended(db_path, sid)


def get_session(db_path, session_id: str) -> dict | None:
    return get_session_extended(db_path, session_id)


def list_sessions(db_path) -> list:
    con = _conn(db_path)
    rows = con.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
    con.close()
    cols = ["id","name","exchange","timeframes","quote_asset","status",
            "created_at","updated_at","notes","entry_logic","entry_mode","pairlist_config",
            "entry_logic_long","entry_logic_short"]
    result = []
    for row in rows:
        d = dict(zip(cols[:len(row)], row))
        d.setdefault("entry_logic", "")
        d.setdefault("entry_mode", "path_b")
        d.setdefault("pairlist_config", "[]")
        d.setdefault("entry_logic_long", "")
        d.setdefault("entry_logic_short", "")
        d["timeframes"] = json.loads(d["timeframes"])
        try:
            d["pairlist_config"] = json.loads(d["pairlist_config"] or "[]")
        except Exception:
            d["pairlist_config"] = []
        result.append(d)
    return result


def update_session_status(db_path, session_id: str, status: str) -> None:
    con = _conn(db_path)
    con.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?",
                [status, datetime.now(timezone.utc), session_id])
    con.close()


def delete_session(db_path, session_id: str) -> None:
    con = _conn(db_path)
    for tbl in ["sessions","pairs","backtest_runs","trades","entry_analysis","tasks","algo_results"]:
        if tbl == "sessions":
            con.execute(f"DELETE FROM {tbl} WHERE id=?", [session_id])
        else:
            con.execute(f"DELETE FROM {tbl} WHERE session_id=?", [session_id])
    con.close()


# ── Pairs ────────────────────────────────────────────────────────────────────

def upsert_pairs(db_path, session_id: str, pairs: list) -> None:
    if not pairs:
        return
    con = _conn(db_path)
    now = datetime.now(timezone.utc)
    # Fetch existing symbols in one query to avoid N+1 round-trips
    existing_rows = con.execute(
        "SELECT symbol, id FROM pairs WHERE session_id=?", [session_id]
    ).fetchall()
    existing = {row[0]: row[1] for row in existing_rows}

    updates, inserts = [], []
    for p in pairs:
        sym = p["symbol"]
        if sym in existing:
            updates.append([p.get("active", True), p.get("volume_24h", 0), session_id, sym])
        else:
            inserts.append([
                str(uuid.uuid4()), session_id, sym,
                p.get("base_asset", ""), p.get("quote_asset", "USDT"),
                p.get("active", True), p.get("volume_24h", 0),
                False, None, None, 0, now
            ])

    if updates:
        con.executemany(
            "UPDATE pairs SET active=?, volume_24h=? WHERE session_id=? AND symbol=?",
            updates,
        )
    if inserts:
        con.executemany("INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", inserts)
    con.close()


def list_pairs(db_path, session_id: str) -> list:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT id, session_id, symbol, base_asset, quote_asset, active,
               volume_24h, excluded, data_start, data_end, candle_count, created_at
        FROM pairs WHERE session_id=? ORDER BY volume_24h DESC
    """, [session_id]).fetchall()
    con.close()
    cols = ["id","session_id","symbol","base_asset","quote_asset","active",
            "volume_24h","excluded","data_start","data_end","candle_count","created_at"]
    return [dict(zip(cols, r)) for r in rows]


def update_pair_data_info(db_path, session_id: str, symbol: str,
                          data_start, data_end, candle_count: int) -> None:
    con = _conn(db_path)
    con.execute("""
        UPDATE pairs SET data_start=?, data_end=?, candle_count=?
        WHERE session_id=? AND symbol=?
    """, [data_start, data_end, candle_count, session_id, symbol])
    con.close()


def set_pair_excluded(db_path, session_id: str, symbol: str, excluded: bool) -> None:
    con = _conn(db_path)
    con.execute("UPDATE pairs SET excluded=? WHERE session_id=? AND symbol=?",
                [excluded, session_id, symbol])
    con.close()


# ── Tasks ────────────────────────────────────────────────────────────────────

def create_task(db_path, session_id: str, task_type: str) -> str:
    tid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    con = _conn(db_path)
    con.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [tid, session_id, task_type, "pending", 0, 0, "", "{}", "", now, now])
    con.close()
    return tid


def update_task(db_path, task_id: str, status: str = None, progress: int = None,
                total: int = None, message: str = None, result: dict = None,
                error: str = None) -> None:
    fields, values = [], []
    now = datetime.now(timezone.utc)
    if status is not None:
        fields.append("status=?"); values.append(status)
    if progress is not None:
        fields.append("progress=?"); values.append(progress)
    if total is not None:
        fields.append("total=?"); values.append(total)
    if message is not None:
        fields.append("message=?"); values.append(message)
    if result is not None:
        fields.append("result=?"); values.append(json.dumps(result))
    if error is not None:
        fields.append("error=?"); values.append(error)
    if not fields:
        return
    fields.append("updated_at=?"); values.append(now)
    values.append(task_id)
    con = _conn(db_path)
    con.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id=?", values)
    con.close()


def get_task(db_path, task_id: str) -> dict | None:
    con = _conn(db_path)
    row = con.execute("SELECT * FROM tasks WHERE id=?", [task_id]).fetchone()
    con.close()
    if not row:
        return None
    cols = ["id","session_id","task_type","status","progress","total",
            "message","result","error","created_at","updated_at"]
    d = dict(zip(cols, row))
    try:
        d["result"] = json.loads(d["result"])
    except Exception:
        d["result"] = {}
    return d


def get_task_status(db_path, task_id: str) -> str | None:
    """Return only the status field of a task by its id. Fast single-field read."""
    con = _conn(db_path)
    row = con.execute("SELECT status FROM tasks WHERE id=?", [task_id]).fetchone()
    con.close()
    return row[0] if row else None


def get_latest_task(db_path, session_id: str, task_type: str) -> dict | None:
    con = _conn(db_path)
    row = con.execute("""
        SELECT * FROM tasks WHERE session_id=? AND task_type=?
        ORDER BY created_at DESC LIMIT 1
    """, [session_id, task_type]).fetchone()
    con.close()
    if not row:
        return None
    cols = ["id","session_id","task_type","status","progress","total",
            "message","result","error","created_at","updated_at"]
    d = dict(zip(cols, row))
    try:
        d["result"] = json.loads(d["result"])
    except Exception:
        d["result"] = {}
    return d


# ── Backtest runs ─────────────────────────────────────────────────────────────

def create_backtest_run(db_path, session_id: str, strategy_code: str, params: dict) -> str:
    rid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    con = _conn(db_path)
    con.execute("""
        INSERT INTO backtest_runs (id, session_id, strategy_code, strategy_params,
            status, created_at) VALUES (?,?,?,?,?,?)
    """, [rid, session_id, strategy_code, json.dumps(params), "pending", now])
    con.close()
    return rid


def update_backtest_run(db_path, run_id: str, **kwargs) -> None:
    allowed = {"status","total_trades","win_rate","profit_factor","sharpe_ratio",
               "max_drawdown","total_return","avg_trade_duration",
               "oos_return","oos_win_rate","error_msg","completed_at"}
    fields, values = [], []
    for k, v in kwargs.items():
        if k in allowed:
            fields.append(f"{k}=?"); values.append(v)
    if not fields:
        return
    values.append(run_id)
    con = _conn(db_path)
    con.execute(f"UPDATE backtest_runs SET {', '.join(fields)} WHERE id=?", values)
    con.close()


def list_backtest_runs(db_path, session_id: str) -> list:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT id, session_id, strategy_params, status, total_trades, win_rate,
               profit_factor, sharpe_ratio, max_drawdown, total_return,
               oos_return, oos_win_rate, created_at, completed_at, error_msg
        FROM backtest_runs WHERE session_id=? ORDER BY created_at DESC
    """, [session_id]).fetchall()
    con.close()
    cols = ["id","session_id","strategy_params","status","total_trades","win_rate",
            "profit_factor","sharpe_ratio","max_drawdown","total_return",
            "oos_return","oos_win_rate","created_at","completed_at","error_msg"]
    result = []
    for r in rows:
        d = dict(zip(cols, r))
        try:
            d["strategy_params"] = json.loads(d["strategy_params"])
        except Exception:
            d["strategy_params"] = {}
        result.append(d)
    return result


def get_backtest_run(db_path, run_id: str) -> dict | None:
    con = _conn(db_path)
    row = con.execute("SELECT * FROM backtest_runs WHERE id=?", [run_id]).fetchone()
    con.close()
    if not row:
        return None
    cols = ["id","session_id","strategy_code","strategy_params","status","total_trades",
            "win_rate","profit_factor","sharpe_ratio","max_drawdown","total_return",
            "avg_trade_duration","oos_return","oos_win_rate","created_at","completed_at","error_msg"]
    d = dict(zip(cols, row))
    try:
        d["strategy_params"] = json.loads(d["strategy_params"])
    except Exception:
        d["strategy_params"] = {}
    return d


# ── Trades ───────────────────────────────────────────────────────────────────

def insert_trades(db_path, trades: list) -> None:
    if not trades:
        return
    con = _conn(db_path)
    con.executemany("""
        INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [[
        t["id"], t["run_id"], t["session_id"], t["symbol"], t["timeframe"],
        t["entry_time"], t.get("exit_time"), t["entry_price"], t.get("exit_price"),
        t.get("direction","long"), t.get("pnl_pct",0), t.get("pnl_abs",0),
        t.get("duration_bars",0), t.get("is_winner",False),
        json.dumps(t.get("entry_signals",{})), t.get("exit_reason","")
    ] for t in trades])
    con.close()


def get_trades(db_path, run_id: str, winners_only: bool = False) -> list:
    con = _conn(db_path)
    q = "SELECT * FROM trades WHERE run_id=?"
    params = [run_id]
    if winners_only:
        q += " AND is_winner=TRUE"
    q += " ORDER BY entry_time"
    rows = con.execute(q, params).fetchall()
    con.close()
    cols = ["id","run_id","session_id","symbol","timeframe","entry_time","exit_time",
            "entry_price","exit_price","direction","pnl_pct","pnl_abs",
            "duration_bars","is_winner","entry_signals","exit_reason"]
    result = []
    for r in rows:
        d = dict(zip(cols, r))
        try:
            d["entry_signals"] = json.loads(d["entry_signals"])
        except Exception:
            d["entry_signals"] = {}
        result.append(d)
    return result


# ── Entry Analysis ────────────────────────────────────────────────────────────

def insert_entry_analysis(db_path, analyses: list) -> None:
    if not analyses:
        return
    con = _conn(db_path)
    now = datetime.now(timezone.utc)
    con.executemany("INSERT INTO entry_analysis VALUES (?,?,?,?,?,?,?,?,?,?,?)", [[
        str(uuid.uuid4()), a["run_id"], a["trade_id"], a["session_id"],
        a["symbol"], a["timeframe"], a["entry_time"],
        json.dumps(a.get("indicator_snapshot",{})),
        json.dumps(a.get("candle_context",{})),
        json.dumps(a.get("pattern_flags",{})),
        now
    ] for a in analyses])
    con.close()


def get_entry_analyses(db_path, run_id: str) -> list:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT * FROM entry_analysis WHERE run_id=? ORDER BY entry_time
    """, [run_id]).fetchall()
    con.close()
    cols = ["id","run_id","trade_id","session_id","symbol","timeframe","entry_time",
            "indicator_snapshot","candle_context","pattern_flags","created_at"]
    result = []
    for r in rows:
        d = dict(zip(cols, r))
        for fld in ["indicator_snapshot","candle_context","pattern_flags"]:
            try:
                d[fld] = json.loads(d[fld])
            except Exception:
                d[fld] = {}
        result.append(d)
    return result


# ── Algo Results ─────────────────────────────────────────────────────────────

def save_algo_result(db_path, session_id: str, run_id: str, rank: int,
                     strategy_name: str, params: dict, rules_description: str,
                     metrics: dict) -> None:
    con = _conn(db_path)
    con.execute("""
        INSERT INTO algo_results
            (id, session_id, run_id, rank, strategy_name, params, rules_description,
             is_return, oos_return, win_rate, sharpe, max_drawdown,
             profit_factor, calmar_ratio, sortino_ratio, expectancy,
             pair_coverage, regime, template, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        str(uuid.uuid4()), session_id, run_id, rank, strategy_name,
        json.dumps(params), rules_description,
        metrics.get("is_return", 0), metrics.get("oos_return", 0),
        metrics.get("win_rate", 0), metrics.get("sharpe", 0),
        metrics.get("max_drawdown", 0),
        metrics.get("profit_factor", 0), metrics.get("calmar_ratio", 0),
        metrics.get("sortino_ratio", 0), metrics.get("expectancy", 0),
        metrics.get("pair_coverage", 0),
        metrics.get("regime", "all"), metrics.get("template", "free"),
        datetime.now(timezone.utc)
    ])
    con.close()


# ── Entry Logic & Pairlist Config ────────────────────────────────────────────

def save_entry_logic(db_path, session_id: str, entry_logic: str, entry_mode: str,
                     entry_logic_long: str = "", entry_logic_short: str = "") -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE sessions SET entry_logic=?, entry_mode=?, entry_logic_long=?, entry_logic_short=?, updated_at=? WHERE id=?",
        [entry_logic, entry_mode, entry_logic_long, entry_logic_short, datetime.now(timezone.utc), session_id]
    )
    con.close()


def save_pairlist_config(db_path, session_id: str, config: list) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE sessions SET pairlist_config=?, updated_at=? WHERE id=?",
        [json.dumps(config), datetime.now(timezone.utc), session_id]
    )
    con.close()


def get_session_extended(db_path, session_id: str) -> dict | None:
    """Like get_session but also returns entry_logic, entry_mode, pairlist_config, entry_logic_long/short."""
    con = _conn(db_path)
    row = con.execute("SELECT * FROM sessions WHERE id=?", [session_id]).fetchone()
    con.close()
    if not row:
        return None
    cols = ["id", "name", "exchange", "timeframes", "quote_asset", "status",
            "created_at", "updated_at", "notes", "entry_logic", "entry_mode", "pairlist_config",
            "entry_logic_long", "entry_logic_short"]
    # Handle tables created before migration (fewer columns)
    d = dict(zip(cols[:len(row)], row))
    d.setdefault("entry_logic", "")
    d.setdefault("entry_mode", "path_b")
    d.setdefault("pairlist_config", "[]")
    d.setdefault("entry_logic_long", "")
    d.setdefault("entry_logic_short", "")
    d["timeframes"] = json.loads(d["timeframes"])
    try:
        d["pairlist_config"] = json.loads(d["pairlist_config"] or "[]")
    except Exception:
        d["pairlist_config"] = []
    return d


# ── Entry Logic Analysis Results ──────────────────────────────────────────────

def save_entry_analysis_result(db_path, session_id: str, result: dict) -> None:
    """Persist the winner/loser indicator discrimination result as a task result."""
    con = _conn(db_path)
    now = datetime.now(timezone.utc)
    tid = str(uuid.uuid4())
    con.execute(
        "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [tid, session_id, "path_a_analysis", "done", 1, 1,
         "Analysis complete", json.dumps(result), "", now, now]
    )
    con.close()
    return tid


def list_algo_results(db_path, session_id: str) -> list:
    con = _conn(db_path)
    cur = con.execute("""
        SELECT * FROM algo_results WHERE session_id=? ORDER BY rank
    """, [session_id])
    col_names = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    con.close()
    result = []
    for r in rows:
        d = dict(zip(col_names, r))
        for k in ["profit_factor","calmar_ratio","sortino_ratio","expectancy","pair_coverage"]:
            d.setdefault(k, 0)
        d.setdefault("regime", "all")
        d.setdefault("template", "free")
        try:
            d["params"] = json.loads(d["params"])
        except Exception:
            d["params"] = {}
        result.append(d)
    return result


def clear_algo_results(db_path, session_id: str) -> None:
    con = _conn(db_path)
    con.execute("DELETE FROM algo_results WHERE session_id=?", [session_id])
    con.close()


# ── Backtest Configs ──────────────────────────────────────────────────────────

_BT_CONFIG_COLS = [
    "id", "session_id", "name", "entry_code", "exit_code", "selected_pairs",
    "initial_capital", "fee_rate", "slippage", "position_sizing", "position_size",
    "kelly_fraction", "atr_risk_pct",
    "sl_mode", "sl_pct", "sl_atr_period", "sl_atr_multiplier",
    "tp_mode", "tp_pct", "tp_atr_period", "tp_atr_multiplier", "tp_rr_ratio",
    "trail_mode", "trail_pct", "trail_atr_period", "trail_atr_multiplier",
    "wfo_splits", "wfo_train_ratio", "created_at", "updated_at",
]


def upsert_backtest_config(db_path, session_id: str, cfg: dict) -> str:
    """Create or update a backtest config. Returns config id."""
    con = _conn(db_path)
    now = datetime.now(timezone.utc)
    cfg_id = cfg.get("id") or str(uuid.uuid4())
    existing = con.execute(
        "SELECT id FROM backtest_configs WHERE id=?", [cfg_id]
    ).fetchone()
    if existing:
        con.execute("""
            UPDATE backtest_configs SET
                name=?, entry_code=?, exit_code=?, selected_pairs=?,
                initial_capital=?, fee_rate=?, slippage=?,
                position_sizing=?, position_size=?, kelly_fraction=?, atr_risk_pct=?,
                sl_mode=?, sl_pct=?, sl_atr_period=?, sl_atr_multiplier=?,
                tp_mode=?, tp_pct=?, tp_atr_period=?, tp_atr_multiplier=?, tp_rr_ratio=?,
                trail_mode=?, trail_pct=?, trail_atr_period=?, trail_atr_multiplier=?,
                wfo_splits=?, wfo_train_ratio=?, updated_at=?
            WHERE id=?
        """, [
            cfg.get("name","Default"), cfg.get("entry_code",""), cfg.get("exit_code",""),
            json.dumps(cfg.get("selected_pairs",[])),
            cfg.get("initial_capital",10000), cfg.get("fee_rate",0.001),
            cfg.get("slippage",0.0005), cfg.get("position_sizing","fixed"),
            cfg.get("position_size",0.1), cfg.get("kelly_fraction",0.25),
            cfg.get("atr_risk_pct",1.0),
            cfg.get("sl_mode","none"), cfg.get("sl_pct",2.0),
            cfg.get("sl_atr_period",14), cfg.get("sl_atr_multiplier",2.0),
            cfg.get("tp_mode","none"), cfg.get("tp_pct",4.0),
            cfg.get("tp_atr_period",14), cfg.get("tp_atr_multiplier",4.0),
            cfg.get("tp_rr_ratio",2.0),
            cfg.get("trail_mode","none"), cfg.get("trail_pct",2.0),
            cfg.get("trail_atr_period",14), cfg.get("trail_atr_multiplier",1.5),
            cfg.get("wfo_splits",5), cfg.get("wfo_train_ratio",0.7),
            now, cfg_id,
        ])
    else:
        con.execute("""
            INSERT INTO backtest_configs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            cfg_id, session_id,
            cfg.get("name","Default"), cfg.get("entry_code",""), cfg.get("exit_code",""),
            json.dumps(cfg.get("selected_pairs",[])),
            cfg.get("initial_capital",10000), cfg.get("fee_rate",0.001),
            cfg.get("slippage",0.0005), cfg.get("position_sizing","fixed"),
            cfg.get("position_size",0.1), cfg.get("kelly_fraction",0.25),
            cfg.get("atr_risk_pct",1.0),
            cfg.get("sl_mode","none"), cfg.get("sl_pct",2.0),
            cfg.get("sl_atr_period",14), cfg.get("sl_atr_multiplier",2.0),
            cfg.get("tp_mode","none"), cfg.get("tp_pct",4.0),
            cfg.get("tp_atr_period",14), cfg.get("tp_atr_multiplier",4.0),
            cfg.get("tp_rr_ratio",2.0),
            cfg.get("trail_mode","none"), cfg.get("trail_pct",2.0),
            cfg.get("trail_atr_period",14), cfg.get("trail_atr_multiplier",1.5),
            cfg.get("wfo_splits",5), cfg.get("wfo_train_ratio",0.7),
            now, now,
        ])
    con.close()
    return cfg_id


def get_backtest_config(db_path, cfg_id: str) -> dict | None:
    con = _conn(db_path)
    row = con.execute("SELECT * FROM backtest_configs WHERE id=?", [cfg_id]).fetchone()
    con.close()
    if not row:
        return None
    d = dict(zip(_BT_CONFIG_COLS[:len(row)], row))
    try:
        d["selected_pairs"] = json.loads(d.get("selected_pairs","[]"))
    except Exception:
        d["selected_pairs"] = []
    return d


def get_latest_backtest_config(db_path, session_id: str) -> dict | None:
    con = _conn(db_path)
    row = con.execute("""
        SELECT * FROM backtest_configs WHERE session_id=?
        ORDER BY updated_at DESC LIMIT 1
    """, [session_id]).fetchone()
    con.close()
    if not row:
        return None
    d = dict(zip(_BT_CONFIG_COLS[:len(row)], row))
    try:
        d["selected_pairs"] = json.loads(d.get("selected_pairs","[]"))
    except Exception:
        d["selected_pairs"] = []
    return d


def list_backtest_configs(db_path, session_id: str) -> list:
    con = _conn(db_path)
    rows = con.execute("""
        SELECT * FROM backtest_configs WHERE session_id=? ORDER BY updated_at DESC
    """, [session_id]).fetchall()
    con.close()
    result = []
    for r in rows:
        d = dict(zip(_BT_CONFIG_COLS[:len(r)], r))
        try:
            d["selected_pairs"] = json.loads(d.get("selected_pairs","[]"))
        except Exception:
            d["selected_pairs"] = []
        result.append(d)
    return result
