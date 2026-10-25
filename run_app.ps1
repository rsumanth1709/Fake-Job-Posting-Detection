# PowerShell runner for JobShield AI (Flask)
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "       Starting JobShield AI Web Application (Flask)...       " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

if (-Not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
} else {
    & .venv\Scripts\Activate.ps1
}

if (-Not (Test-Path "models\model.joblib")) {
    Write-Host "[INFO] Training models on dataset..." -ForegroundColor Yellow
    python train_model.py
}

Write-Host "[INFO] Starting Flask Web App on http://127.0.0.1:5000" -ForegroundColor Green
python server.py
