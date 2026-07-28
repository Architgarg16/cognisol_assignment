$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDirectory = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating Python virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $venvDirectory
        if ($LASTEXITCODE -ne 0) {
            & py -3 -m venv $venvDirectory
        }
    } else {
        & python -m venv $venvDirectory
    }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment creation failed."
}

Write-Host "Installing dependencies and the local package..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e $projectRoot

Write-Host ""
Write-Host "Setup complete."
Write-Host "In PyCharm select this interpreter:"
Write-Host $venvPython
Write-Host ""
Write-Host "Then run the shared configuration: Web UI - Streamlit"
