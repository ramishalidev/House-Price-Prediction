@echo off
echo ========================================
echo    House Price Prediction System
echo ========================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt -q

:: Start FastAPI server in background
echo [2/3] Starting FastAPI server on port 8000...
start /B python -m uvicorn api:app --host 0.0.0.0 --port 8000

:: Wait a moment for API to start
timeout /t 3 /nobreak >nul

:: Start Streamlit
echo [3/3] Starting Streamlit UI...
echo.
echo ========================================
echo    Open in browser: http://localhost:8501
echo    API docs: http://localhost:8000/docs
echo ========================================
echo.

streamlit run app.py

pause
