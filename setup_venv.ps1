Param()

# Create and activate a virtual environment, then install requirements
$venvPath = Join-Path $PSScriptRoot 'venv'
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

Write-Host "Activating venv..."
& (Join-Path $venvPath 'Scripts\Activate.ps1')

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

Write-Host "Done. To activate later: .\venv\Scripts\Activate.ps1"
