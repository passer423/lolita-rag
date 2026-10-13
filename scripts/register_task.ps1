# Thin ASCII-only wrapper, kept so that older documented commands still work.
#
# The real implementation is scripts\register_task.py.
# Reason for the move: this machine only has Windows PowerShell 5.1, which
# decodes BOM-less UTF-8 .ps1 files as ANSI. Chinese text then turns into
# mojibake that swallows quotes and breaks parsing - and the BOM is lost again
# by any later edit. Python + schtasks.exe has no such trap.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1 -Time 03:30
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1 -Status
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1 -RunNow
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1 -Unregister
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1 -DryRun

param(
    [string]$Time = "",
    [string]$Python = "",
    [switch]$Status,
    [switch]$Unregister,
    [switch]$RunNow,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$entry = Join-Path $root "scripts\register_task.py"
if (-not (Test-Path $entry)) { throw "not found: $entry" }

$pythonExe = $Python
if (-not $pythonExe) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "python not found on PATH; pass -Python <path to python.exe>" }
    $pythonExe = $cmd.Source
}

$arguments = @($entry)
if ($Time) { $arguments += @("--time", $Time) }
if ($Status) { $arguments += "--status" }
if ($Unregister) { $arguments += "--unregister" }
if ($RunNow) { $arguments += "--run-now" }
if ($DryRun) { $arguments += "--dry-run" }

& $pythonExe @arguments
exit $LASTEXITCODE
