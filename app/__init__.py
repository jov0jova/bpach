import logging
import os
from pathlib import Path

from flask import Flask

from .config import Config
from .models import init_db


def create_app(config_class=Config) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="../static")
    app.config.from_object(config_class)

    # Ensure directories exist
    for d in [config_class.DATA_DIR, config_class.PARQUET_DIR,
              config_class.SESSIONS_DIR, Path("logs")]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config_class.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log", encoding="utf-8"),
        ]
    )

    # Initialize database
    init_db(config_class.DB_PATH)

    # Custom Jinja2 filters
    @app.template_filter("first_value")
    def first_value_filter(d):
        """Return the first value of a dict."""
        if not d:
            return {}
        return next(iter(d.values()), {})

    # Register blueprints
    from .routes.main import bp as main_bp
    from .routes.sessions import bp as sessions_bp
    from .routes.data import bp as data_bp
    from .routes.pairlist import bp as pairlist_bp
    from .routes.entry_logic import bp as entry_logic_bp
    from .routes.strategy import bp as strategy_bp
    from .routes.indicators import bp as indicators_bp
    from .routes.backtest import bp as backtest_bp
    from .routes.analysis import bp as analysis_bp
    from .routes.algofinder import bp as algofinder_bp
    from .routes.ic_analysis import bp as ic_analysis_bp
    from .routes.indicators_lib import bp as indicators_lib_bp
    from .routes.charts import bp as charts_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(sessions_bp, url_prefix="/sessions")
    app.register_blueprint(data_bp, url_prefix="/data")
    app.register_blueprint(pairlist_bp, url_prefix="/pairlist")
    app.register_blueprint(entry_logic_bp, url_prefix="/entry_logic")
    app.register_blueprint(strategy_bp, url_prefix="/strategy")
    app.register_blueprint(indicators_bp, url_prefix="/indicators")
    app.register_blueprint(backtest_bp, url_prefix="/backtest")
    app.register_blueprint(analysis_bp, url_prefix="/analysis")
    app.register_blueprint(algofinder_bp, url_prefix="/algofinder")
    app.register_blueprint(ic_analysis_bp, url_prefix="/ic_analysis")
    app.register_blueprint(indicators_lib_bp, url_prefix="/indicators_lib")
    app.register_blueprint(charts_bp, url_prefix="/charts")

    return app
