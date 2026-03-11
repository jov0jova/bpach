@echo off
echo ============================================
echo  CryptoAlgoFinder - Starting...
echo ============================================
echo.

if not exist venv (
    echo ERROR: Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist .env (
    copy .env.example .env
)

echo Starting Flask server on http://localhost:5000
echo Press Ctrl+C to stop.
echo.

start "" http://localhost:5000
python run.py
