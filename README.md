# CryptoAlgoFinder

A local Windows desktop Flask application for discovering profitable crypto trading strategies through automated backtesting and reverse-engineering of winning trades.

## Quick Start

```
git clone <repo>
cd CryptoAlgoFinder
setup.bat      # Creates venv, installs deps
run.bat        # Starts Flask server and opens browser
```

## 8-Phase Workflow

| # | Phase | Description |
|---|-------|-------------|
| 2 | **Download Data** | Fetch OHLCV history from exchange via CCXT |
| 3 | **Pairlist** | Filter and manage trading pairs |
| 4 | **Strategy** | Define entry/exit rules in Python |
| 5 | **Add Indicators** | Enrich data with ~100 TA indicators |
| 6 | **Backtest** | Walk-forward optimised backtest |
| 7 | **Analysis** | Reverse-engineer winning trades |
| 8 | **Algo Finder** | Optuna search for best parameter combos |

## Requirements

- Python 3.11+
- Windows 10/11
- Internet connection (for initial data download only)

## Technology Stack

- **Flask 3.0** – Web framework
- **DuckDB** – Local database
- **pandas-ta** – Technical indicators
- **vectorbt** – Fast backtesting
- **Optuna** – Hyperparameter optimization
- **CCXT** – Exchange data
- **Plotly** – Interactive charts
- **HTMX** – Live progress updates

## Architecture

All data is stored locally:
- `data/app.db` – DuckDB database (sessions, trades, analysis)
- `data/parquet/` – OHLCV + indicator data per symbol/timeframe

No authentication, no cloud, no Docker required.
