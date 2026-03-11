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

    # Threading
    MAX_WORKERS = 2

    # Backtest defaults
    DEFAULT_INITIAL_CAPITAL = 10_000.0
    DEFAULT_FEE_RATE = 0.001       # 0.1% taker fee
    DEFAULT_SLIPPAGE = 0.0005      # 0.05%
    DEFAULT_POSITION_SIZE = 0.1    # 10% of capital per trade

    # Walk-forward
    WFO_SPLITS = 5
    WFO_TRAIN_RATIO = 0.7

    # Optuna
    OPTUNA_TRIALS = 50
    OPTUNA_TIMEOUT = 300  # seconds

    # CCXT download
    DOWNLOAD_LIMIT_PER_REQUEST = 1000
    MAX_PAIRS = 200
