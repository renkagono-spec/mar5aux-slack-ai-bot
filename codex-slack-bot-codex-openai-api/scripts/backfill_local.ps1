param(
    [Parameter(Mandatory=$true)]
    [string]$ChannelId,

    [int]$MaxMessages = 1000
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
    Write-Error ".env is missing. Copy .env.example to .env and fill SLACK_BOT_TOKEN, OPENAI_API_KEY, and DATABASE_URL."
}

& $Python (Join-Path $Root "scripts\backfill.py") $ChannelId --max-messages $MaxMessages
