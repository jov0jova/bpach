import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-me-in-production")
    DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
    DB_PATH = DATA_DIR / "app.db"
    PARQUET_DIR = DATA_DIR / "parquet"
    SESSIONS_DIR = DATA_DIR / "sessions"
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

    # Threading: allow up to cpu_count concurrent background tasks (min 2)
    import os as _os
    MAX_WORKERS = max(2, _os.cpu_count() or 2)

    # ── Backtest defaults ─────────────────────────────────────────────────────
    DEFAULT_INITIAL_CAPITAL = 10_000.0
    DEFAULT_FEE_RATE = 0.001       # 0.1% taker fee
    DEFAULT_SLIPPAGE = 0.0005      # 0.05%
    DEFAULT_POSITION_SIZE = 0.1    # 10% of capital per trade

    # Stop loss modes: "none" | "fixed" | "atr"
    DEFAULT_SL_MODE = "none"
    DEFAULT_SL_PCT = 2.0            # fixed SL %
    DEFAULT_SL_ATR_PERIOD = 14
    DEFAULT_SL_ATR_MULTIPLIER = 2.0

    # Take profit modes: "none" | "fixed" | "atr" | "rr"  (rr = risk:reward ratio)
    DEFAULT_TP_MODE = "none"
    DEFAULT_TP_PCT = 4.0
    DEFAULT_TP_ATR_PERIOD = 14
    DEFAULT_TP_ATR_MULTIPLIER = 4.0
    DEFAULT_TP_RR_RATIO = 2.0       # take profit at 2× the stop loss distance

    # Trailing stop modes: "none" | "fixed" | "atr"
    DEFAULT_TRAIL_MODE = "none"
    DEFAULT_TRAIL_PCT = 2.0
    DEFAULT_TRAIL_ATR_PERIOD = 14
    DEFAULT_TRAIL_ATR_MULTIPLIER = 1.5

    # Position sizing modes: "fixed" | "kelly" | "atr_risk"
    DEFAULT_POSITION_SIZING = "fixed"
    DEFAULT_KELLY_FRACTION = 0.25   # fractional Kelly (safety)
    DEFAULT_ATR_RISK_PCT = 1.0      # risk 1% of capital per ATR unit

    # Walk-forward
    WFO_SPLITS = 5
    WFO_TRAIN_RATIO = 0.7

    # ── Optuna / Algofinder ───────────────────────────────────────────────────
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 600            # seconds

    # Algofinder search mode: "single" | "multi_objective"
    ALGOFINDER_MODE = "multi_objective"

    # Minimum OOS trades to consider a strategy valid
    ALGOFINDER_MIN_OOS_TRADES = 5

    # Cross-pair generalization: require strategy to work on this fraction of pairs
    ALGOFINDER_MIN_PAIR_COVERAGE = 0.3

    # ── CCXT download ─────────────────────────────────────────────────────────
    DOWNLOAD_LIMIT_PER_REQUEST = 1000
    MAX_PAIRS = 200
