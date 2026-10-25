@echo off
title JobShield AI - Fake Job Posting Detection Web App
echo ==============================================================
echo       Starting JobShield AI Web Application (Flask)...
echo ==============================================================
echo.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [INFO] Installing required dependencies...
    pip install -r requirements.txt
) ELSE (
    call .venv\Scripts\activate.bat
)

IF NOT EXIST "models\model.joblib" (
    echo [INFO] Trained model not found. Training model now on dataset...
    python train_model.py
)

echo [INFO] Launching Flask Web App on http://127.0.0.1:5000...
python server.py

pause
