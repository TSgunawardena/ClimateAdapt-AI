$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install 64-bit Python 3.12 first."
}

$python312 = py -3.12 -c "import sys; print(sys.executable)" 2>$null
if (-not $python312) {
    throw "Python 3.12 was not found. Install it before continuing."
}

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: python .\scripts\01_download_ncc_labels.py"
