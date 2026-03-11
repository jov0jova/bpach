"""
seed_sample_session.py
----------------------
Creates a sample session with fake pairs in the local DuckDB database so
you can test the UI without running a real exchange download.

Usage:
    python seed_sample_session.py

The script prints the session ID and the URL to open in your browser.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

DB_PATH = Path("data/app.db")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    # Ensure tables exist (minimal subset needed here)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, exchange TEXT NOT NULL DEFAULT 'binance',
            timeframes TEXT NOT NULL DEFAULT '["15m","1h","4h"]', quote_asset TEXT NOT NULL DEFAULT 'USDT',
            status TEXT NOT NULL DEFAULT 'created', created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL, notes TEXT DEFAULT '')
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS pairs (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, symbol TEXT NOT NULL,
            base_asset TEXT NOT NULL, quote_asset TEXT NOT NULL, active BOOLEAN NOT NULL DEFAULT TRUE,
            volume_24h DOUBLE DEFAULT 0, excluded BOOLEAN NOT NULL DEFAULT FALSE,
            data_start TIMESTAMP, data_end TIMESTAMP, candle_count INTEGER DEFAULT 0,
            created_at TIMESTAMP NOT NULL)
    """)

    # ── Create session ────────────────────────────────────────────────────────
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    con.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
        [
            sid,
            "Sample BTC/USDT Session",
            "binance",
            json.dumps(["15m", "1h", "4h"]),
            "USDT",
            "created",
            now,
            now,
            "Auto-generated sample session for testing the UI.",
        ],
    )

    # ── Seed pairs ────────────────────────────────────────────────────────────
    pairs = [
        ("BTC/USDT", "BTC", 9_500_000_000),
        ("ETH/USDT", "ETH", 4_200_000_000),
        ("BNB/USDT", "BNB", 1_100_000_000),
        ("SOL/USDT", "SOL", 850_000_000),
        ("XRP/USDT", "XRP", 620_000_000),
        ("ADA/USDT", "ADA", 310_000_000),
        ("AVAX/USDT", "AVAX", 290_000_000),
        ("DOGE/USDT", "DOGE", 270_000_000),
        ("LINK/USDT", "LINK", 210_000_000),
        ("DOT/USDT", "DOT", 180_000_000),
    ]

    data_end = now.replace(tzinfo=timezone.utc)
    data_start = data_end - timedelta(days=365)

    for symbol, base, vol in pairs:
        con.execute(
            "INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                str(uuid.uuid4()),
                sid,
                symbol,
                base,
                "USDT",
                True,        # active
                vol,         # volume_24h
                False,       # excluded
                data_start,  # data_start
                data_end,    # data_end
                35040,       # candle_count (1 year of 15m candles)
                now,         # created_at
            ],
        )

    con.close()

    print(f"\n✓ Sample session created!")
    print(f"  Session ID : {sid}")
    print(f"  Name       : Sample BTC/USDT Session")
    print(f"  Exchange   : binance")
    print(f"  Timeframes : 15m, 1h, 4h")
    print(f"  Pairs      : {len(pairs)}")
    print(f"\n  Open in browser:")
    print(f"    http://127.0.0.1:5000/data/{sid}")
    print(f"    http://127.0.0.1:5000/sessions\n")


if __name__ == "__main__":
    main()
