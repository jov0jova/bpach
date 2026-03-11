import os
import sys
from pathlib import Path

# Ensure data/log directories exist before anything else
for d in ["data", "data/sessions", "data/parquet", "logs"]:
    Path(d).mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="127.0.0.1",
        port=port,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
        use_reloader=False,  # Reloader conflicts with background threads
    )
