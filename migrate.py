"""
Database migration script for CryptoAlgoFinder.

Runs all schema migrations in order, reporting each step.
Safe to run multiple times (idempotent).

Usage:
    python migrate.py
"""
import sys
from pathlib import Path

# Ensure data directory exists before anything else
Path("data").mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

import duckdb

DB_PATH = Path("data/app.db")


def _conn():
    return duckdb.connect(str(DB_PATH))


def _col_exists(con, table: str, column: str) -> bool:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = ?",
        [table, column]
    ).fetchall()
    return len(rows) > 0


def _table_exists(con, table: str) -> bool:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
        [table]
    ).fetchall()
    return len(rows) > 0


def step(label: str, fn):
    """Run a migration step and report its result."""
    try:
        result = fn()
        status = result if isinstance(result, str) else "OK"
        print(f"  [OK] {label}: {status}")
    except Exception as e:
        print(f"  [!!] {label}: {e}", file=sys.stderr)
        raise


def run_migrations():
    print()
    print("=" * 52)
    print("  CryptoAlgoFinder — Database Migration")
    print("=" * 52)
    print(f"  DB: {DB_PATH.resolve()}")
    print()

    con = _conn()

    # ── 1. Core tables ────────────────────────────────────────────────────────
    print("[1] Core tables")

    def create_sessions():
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                exchange        TEXT NOT NULL DEFAULT 'binance',
                timeframes      TEXT NOT NULL DEFAULT '["15m","1h","4h"]',
                quote_asset     TEXT NOT NULL DEFAULT 'USDT',
                status          TEXT NOT NULL DEFAULT 'created',
                created_at      TIMESTAMP NOT NULL,
                updated_at      TIMESTAMP NOT NULL,
                notes           TEXT DEFAULT '',
                entry_logic     TEXT DEFAULT '',
                entry_mode      TEXT NOT NULL DEFAULT 'path_b',
                pairlist_config TEXT DEFAULT '[]'
            )
        """)
        return "sessions"

    def create_pairs():
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
        return "pairs"

    def create_tasks():
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
        return "tasks"

    def create_backtest_runs():
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
        return "backtest_runs"

    def create_trades():
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
        return "trades"

    def create_entry_analysis():
        con.execute("""
            CREATE TABLE IF NOT EXISTS entry_analysis (
                id                  TEXT PRIMARY KEY,
                run_id              TEXT NOT NULL,
                trade_id            TEXT NOT NULL,
                session_id          TEXT NOT NULL,
                symbol              TEXT NOT NULL,
                timeframe           TEXT NOT NULL,
                entry_time          TIMESTAMP NOT NULL,
                indicator_snapshot  TEXT NOT NULL DEFAULT '{}',
                candle_context      TEXT NOT NULL DEFAULT '{}',
                pattern_flags       TEXT NOT NULL DEFAULT '{}',
                created_at          TIMESTAMP NOT NULL
            )
        """)
        return "entry_analysis"

    def create_algo_results():
        con.execute("""
            CREATE TABLE IF NOT EXISTS algo_results (
                id                TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL,
                run_id            TEXT,
                rank              INTEGER NOT NULL,
                strategy_name     TEXT NOT NULL,
                params            TEXT NOT NULL DEFAULT '{}',
                rules_description TEXT DEFAULT '',
                is_return         DOUBLE DEFAULT 0,
                oos_return        DOUBLE DEFAULT 0,
                win_rate          DOUBLE DEFAULT 0,
                sharpe            DOUBLE DEFAULT 0,
                max_drawdown      DOUBLE DEFAULT 0,
                created_at        TIMESTAMP NOT NULL
            )
        """)
        return "algo_results"

    step("sessions table",       create_sessions)
    step("pairs table",          create_pairs)
    step("tasks table",          create_tasks)
    step("backtest_runs table",  create_backtest_runs)
    step("trades table",         create_trades)
    step("entry_analysis table", create_entry_analysis)
    step("algo_results table",   create_algo_results)

    # ── 2. Column migrations (ADD IF MISSING) ─────────────────────────────────
    print()
    print("[2] Column migrations")

    column_migrations = [
        # (table, column, typedef, description)
        ("sessions", "entry_logic",     "TEXT DEFAULT ''",              "entry logic storage"),
        ("sessions", "entry_mode",      "TEXT DEFAULT 'path_b'",        "path A / path B mode flag"),
        ("sessions", "pairlist_config", "TEXT DEFAULT '[]'",            "pairlist configuration JSON"),
        ("sessions", "notes",           "TEXT DEFAULT ''",              "session notes field"),
    ]

    for table, col, typedef, desc in column_migrations:
        def _add_col(t=table, c=col, td=typedef, d=desc):
            if not _col_exists(con, t, c):
                con.execute(f"ALTER TABLE {t} ADD COLUMN {c} {td}")
                return f"added column '{c}' to {t} ({d})"
            return f"'{c}' already present in {t}"
        step(f"{table}.{col}", _add_col)

    # ── 3. Data directory structure ───────────────────────────────────────────
    print()
    print("[3] Data directories")

    dirs = ["data", "data/sessions", "data/parquet", "logs"]

    def create_dirs():
        created = []
        for d in dirs:
            p = Path(d)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                created.append(d)
        return f"created {created}" if created else "all exist"

    step("data directories", create_dirs)

    con.close()

    print()
    print("=" * 52)
    print("  Migration complete — ready to start.")
    print("=" * 52)
    print()


if __name__ == "__main__":
    try:
        run_migrations()
    except Exception as e:
        print(f"\nMigration FAILED: {e}", file=sys.stderr)
        sys.exit(1)
