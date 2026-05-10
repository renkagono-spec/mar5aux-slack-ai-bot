param(
    [string[]]$Prefix = @(),
    [string[]]$Contains = @(),
    [string[]]$ChannelId = @(),
    [switch]$All,
    [switch]$Execute,
    [int]$MaxMessages = 1000,
    [string]$Types = "public_channel"
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
    Write-Error ".env is missing. Fill SLACK_BOT_TOKEN, OPENAI_API_KEY, and DATABASE_URL before backfilling."
}

$ArgsList = @("--max-messages", "$MaxMessages", "--types", "$Types")
foreach ($Item in $Prefix) {
    $ArgsList += "--prefix"
    $ArgsList += $Item
}
foreach ($Item in $Contains) {
    $ArgsList += "--contains"
    $ArgsList += $Item
}
foreach ($Item in $ChannelId) {
    $ArgsList += "--channel-id"
    $ArgsList += $Item
}
if ($All) {
    $ArgsList += "--all"
}
if ($Execute) {
    $ArgsList += "--execute"
}

& $Python (Join-Path $Root "scripts\backfill_joined_channels.py") @ArgsList
