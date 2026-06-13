# Windows Task Scheduler wrapper: daily invoice payment-due reminder.
# Runs invoice_reminder.py --post and appends output to logs/invoice_reminder.log.
$ErrorActionPreference = "Continue"
$proj = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $proj
$logDir = Join-Path $proj "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir "invoice_reminder.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$py = Join-Path $proj ".venv\Scripts\python.exe"
# Read python's UTF-8 stdout correctly (default console is cp932 on JP Windows).
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = & $py -X utf8 "scripts\invoice_reminder.py" --post 2>&1 | Out-String
"===== $ts =====`n$out" | Out-File -Append -Encoding utf8 $log
