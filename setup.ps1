# setup.ps1 - HaloCheck one-time setup for Windows 11
# Usage (PowerShell):
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================="
Write-Host "  HaloCheck Setup - Windows 11"
Write-Host "========================================="
Write-Host ""

# 1. Python version check
Write-Host "[1/6] Checking Python version..."
$pyVersion = py --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAIL: Python launcher (py) not found."
    Write-Host "  Install Python 3.11 from https://python.org"
    Write-Host "  Make sure to check 'Add Python to PATH' during install."
    exit 1
}
Write-Host "  Found: $pyVersion"
Write-Host "  OK"

# 2. Virtual environment
Write-Host ""
Write-Host "[2/6] Creating virtual environment..."
py -m venv venv
.\venv\Scripts\Activate.ps1
Write-Host "  OK - venv created and activated"
Write-Host "  (Each session: run .\venv\Scripts\Activate.ps1 before working)"

# 3. Core dependencies
Write-Host ""
Write-Host "[3/6] Installing Python packages..."
py -m pip install --upgrade pip --quiet
py -m pip install -r requirements.txt --quiet
Write-Host "  OK"

# 4. scispaCy clinical model
Write-Host ""
Write-Host "[4/6] Installing scispaCy clinical model (~800MB)..."
py -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
Write-Host "  OK"

# 5. Ollama check
Write-Host ""
Write-Host "[5/6] Checking Ollama..."
$ollamaCheck = Get-Command ollama -ErrorAction SilentlyContinue
if ($null -eq $ollamaCheck) {
    Write-Host "  FAIL: Ollama not found in PATH."
    Write-Host "  Install from https://ollama.com/download then re-run this script."
    exit 1
}
Write-Host "  OK - Ollama found"

$modelList = ollama list 2>&1
if ($modelList -notmatch "gemma2") {
    Write-Host "  gemma2 not found - pulling now (~5GB, this will take a while)..."
    ollama pull gemma2
}
Write-Host "  OK - gemma2 ready"

# 6. Verification
Write-Host ""
Write-Host "[6/6] Running verification checks..."
python -c "import spacy; spacy.load('en_core_sci_lg'); print('  OK: scispaCy en_core_sci_lg')"
python -c "import medspacy; print('  OK: medspaCy')"
python -c "import transformers; print('  OK: transformers')"
python -c "import streamlit; print('  OK: streamlit')"
python -c "import ollama; print('  OK: ollama package')"

Write-Host ""
Write-Host "========================================="
Write-Host "  Setup complete."
Write-Host "========================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. .\venv\Scripts\Activate.ps1"
Write-Host "  2. Open a NEW terminal and run: ollama serve"
Write-Host "  3. python scripts\scraper.py"
Write-Host ""