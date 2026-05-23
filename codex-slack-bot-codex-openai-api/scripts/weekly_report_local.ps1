param(
    [int]$Days = 7,
    [int]$MinChannelMessages = 4,
    [int]$MaxMessagesPerChannel = 80,
    [switch]$Post,
    [string]$Channel = ""
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
    Write-Error ".env is missing. Fill SLACK_BOT_TOKEN, OPENAI_API_KEY, and DATABASE_URL before running."
}

$env:PYTHONIOENCODING = "utf-8"

$ArgsList = @(
    "--days", "$Days",
    "--min-channel-messages", "$MinChannelMessages",
    "--max-messages-per-channel", "$MaxMessagesPerChannel"
)
if ($Post) {
    $ArgsList += "--post"
}
if ($Channel -ne "") {
    $ArgsList += "--channel"
    $ArgsList += $Channel
}

& $Python (Join-Path $Root "scripts\weekly_report.py") @ArgsList
