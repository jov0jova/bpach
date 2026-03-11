@echo off
echo ============================================
echo  CryptoAlgoFinder - Setup
echo ============================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.11+ and add it to PATH.
    pause
    exit /b 1
)

python --version

echo.
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Creating required directories...
if not exist data mkdir data
if not exist data\sessions mkdir data\sessions
if not exist data\parquet mkdir data\parquet
if not exist logs mkdir logs

echo.
echo Copying .env template...
if not exist .env (
    copy .env.example .env
    echo Please edit .env and set a strong FLASK_SECRET_KEY before first run.
)

echo.
echo ============================================
echo  Setup complete! Run run.bat to start.
echo ============================================
pause
