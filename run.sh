#!/bin/bash
echo "========================================"
echo "   House Price Prediction System"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    exit 1
fi

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

# Start FastAPI server in background
echo "[2/3] Starting FastAPI server on port 8000..."
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start Streamlit
echo "[3/3] Starting Streamlit UI..."
echo
echo "========================================"
echo "   Open in browser: http://localhost:8501"
echo "   API docs: http://localhost:8000/docs"
echo "========================================"
echo

streamlit run app.py

# Cleanup
kill $API_PID 2>/dev/null
