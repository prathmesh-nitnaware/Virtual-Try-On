param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $Root ".venv-vton"
$Requirements = Join-Path $Root "requirements-vton.txt"

if (-not (Test-Path $Requirements)) {
    throw "requirements-vton.txt not found at $Requirements"
}

Write-Host "Creating VTON virtual environment at: $VenvPath"
& $PythonExe -m venv $VenvPath

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "VTON venv python not found at $VenvPython"
}

Write-Host "Installing dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r $Requirements

Write-Host "VTON environment ready."
Write-Host "Use: $VenvPython test_pipeline.py"
