<#
.SYNOPSIS
    CANN_OpHelper dependency bootstrap (Windows / PowerShell).

.DESCRIPTION
    Installs this project (editable mode + dev extras) into the EXISTING
    workspace virtual environment "penv" which lives at <workspace>\penv
    (NO leading dot; configured with the embedded Python 3.14.5 + pip).

    IMPORTANT: the project REUSES the workspace-root venv "penv".
    This script NEVER creates a new ".penv" environment. The pip install
    is executed only when the user runs this script.

.PARAMETER VenvPython
    Optional: full path to the venv python.exe.
    When omitted, the script looks for "<workspace root>\penv\Scripts\python.exe"
    (workspace root = parent folder of the project).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap_venv.ps1
#>
[CmdletBinding()]
param(
    [string]$VenvPython = ""
)

$ErrorActionPreference = "Stop"

# Project root = folder that contains the scripts\ subfolder
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Workspace root = parent of the project folder; venv "penv" lives here
$WorkspaceRoot = Split-Path -Parent $ProjectRoot

# 1) Resolve the venv interpreter (never create a new venv)
$Py = ""
if ($VenvPython -and (Test-Path $VenvPython)) {
    $Py = $VenvPython
} else {
    $DefaultVenv = Join-Path $WorkspaceRoot "penv\Scripts\python.exe"
    if (Test-Path $DefaultVenv) { $Py = $DefaultVenv }
}

if (-not $Py) {
    Write-Host "Could not locate the existing venv interpreter." -ForegroundColor Red
    Write-Host "Expected: <workspace root>\penv\Scripts\python.exe" -ForegroundColor Yellow
    Write-Host "or pass -VenvPython <full path to venv python.exe>." -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/2] venv python: $Py" -ForegroundColor Cyan
& $Py --version
if ($LASTEXITCODE -ne 0) { Write-Error "The venv python could not run."; exit 1 }

# 2) Install this project into the existing venv (editable + dev extras)
Write-Host "[2/2] pip install -e '.[dev]' ..." -ForegroundColor Cyan
Push-Location $ProjectRoot
try {
    & $Py -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed (exit $LASTEXITCODE)."; exit 1 }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done. venv python: $Py" -ForegroundColor Green
