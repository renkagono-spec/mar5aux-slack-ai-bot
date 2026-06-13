# Windows Task Scheduler wrapper: weekly report (continues from the last one).
# Runs weekly_report.py --since-last --post and logs to logs/weekly_report.log.
$ErrorActionPreference = "Continue"
$proj = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $proj
$logDir = Join-Path $proj "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir "weekly_report.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$py = Join-Path $proj ".venv\Scripts\python.exe"
# Read python's UTF-8 stdout correctly (default console is cp932 on JP Windows).
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = & $py -X utf8 "scripts\weekly_report.py" --since-last --post --channel C0B48MNK9J7 2>&1 | Out-String
"===== $ts =====`n$out" | Out-File -Append -Encoding utf8 $log
