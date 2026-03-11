"""Thin wrappers around models to provide app-context-aware DB access."""
from flask import current_app
from pathlib import Path

from .. import models as m


def db_path() -> Path:
    return current_app.config["DB_PATH"]


# Re-export all model functions bound to the current app's db_path
def create_session(name, exchange, timeframes, quote_asset):
    return m.create_session(db_path(), name, exchange, timeframes, quote_asset)

def get_session(session_id):
    return m.get_session(db_path(), session_id)

def list_sessions():
    return m.list_sessions(db_path())

def update_session_status(session_id, status):
    return m.update_session_status(db_path(), session_id, status)

def delete_session(session_id):
    return m.delete_session(db_path(), session_id)

def upsert_pairs(session_id, pairs):
    return m.upsert_pairs(db_path(), session_id, pairs)

def list_pairs(session_id):
    return m.list_pairs(db_path(), session_id)

def update_pair_data_info(session_id, symbol, data_start, data_end, candle_count):
    return m.update_pair_data_info(db_path(), session_id, symbol, data_start, data_end, candle_count)

def set_pair_excluded(session_id, symbol, excluded):
    return m.set_pair_excluded(db_path(), session_id, symbol, excluded)

def create_task(session_id, task_type):
    return m.create_task(db_path(), session_id, task_type)

def update_task(task_id, **kwargs):
    return m.update_task(db_path(), task_id, **kwargs)

def get_task(task_id):
    return m.get_task(db_path(), task_id)

def get_latest_task(session_id, task_type):
    return m.get_latest_task(db_path(), session_id, task_type)

def create_backtest_run(session_id, strategy_code, params):
    return m.create_backtest_run(db_path(), session_id, strategy_code, params)

def update_backtest_run(run_id, **kwargs):
    return m.update_backtest_run(db_path(), run_id, **kwargs)

def list_backtest_runs(session_id):
    return m.list_backtest_runs(db_path(), session_id)

def get_backtest_run(run_id):
    return m.get_backtest_run(db_path(), run_id)

def insert_trades(trades):
    return m.insert_trades(db_path(), trades)

def get_trades(run_id, winners_only=False):
    return m.get_trades(db_path(), run_id, winners_only)

def insert_entry_analysis(analyses):
    return m.insert_entry_analysis(db_path(), analyses)

def get_entry_analyses(run_id):
    return m.get_entry_analyses(db_path(), run_id)

def save_algo_result(session_id, run_id, rank, strategy_name, params, rules_description, metrics):
    return m.save_algo_result(db_path(), session_id, run_id, rank, strategy_name,
                              params, rules_description, metrics)

def list_algo_results(session_id):
    return m.list_algo_results(db_path(), session_id)

def save_entry_logic(session_id, entry_logic, entry_mode):
    return m.save_entry_logic(db_path(), session_id, entry_logic, entry_mode)

def save_pairlist_config(session_id, config):
    return m.save_pairlist_config(db_path(), session_id, config)

def get_session_extended(session_id):
    return m.get_session_extended(db_path(), session_id)

def save_entry_analysis_result(session_id, result):
    return m.save_entry_analysis_result(db_path(), session_id, result)
