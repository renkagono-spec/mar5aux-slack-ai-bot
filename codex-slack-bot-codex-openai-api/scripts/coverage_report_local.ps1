param(
    [int]$Limit = 50,
    [switch]$CheckSlack
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$EnvFile = Join-Path $Root ".env"

if (!(Test-Path $Python)) {
    Write-Host "Creating local Python environment..."
    python -m venv (Join-Path $Root ".venv")
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r (Join-Path $Root "requirements.txt")
}

if (!(Test-Path $EnvFile)) {
    Write-Error ".env is missing. Fill DATABASE_URL before checking coverage."
}

$ArgsList = @("--limit", "$Limit")
if ($CheckSlack) {
    $ArgsList += "--check-slack"
}

& $Python (Join-Path $Root "scripts\coverage_report.py") @ArgsList
