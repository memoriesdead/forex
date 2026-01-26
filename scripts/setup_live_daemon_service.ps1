# Setup Forex 24/7 Data Collection as Windows Task
# Run as Administrator: powershell -ExecutionPolicy Bypass -File scripts\setup_live_daemon_service.ps1
#
# This creates a scheduled task that:
# - Runs the full coverage trading bot (TrueFX + IB Gateway + OANDA)
# - Auto-restarts every minute if it crashes
# - Starts at system boot
# - Runs 24/7 for continuous data collection

$taskName = "ForexFullCoverage24x7"
$taskPath = "C:\Users\kevin\forex\forex"
$pythonPath = "C:\Users\kevin\forex\forex\venv\Scripts\python.exe"
$scriptPath = "C:\Users\kevin\forex\forex\scripts\hft_trading_bot.py"
$scriptArgs = "$scriptPath --mode paper --full-coverage --capital 100 --multi-source --online-learning"

# Remove existing task if exists
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create action (use full args for full coverage)
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $scriptArgs -WorkingDirectory $taskPath

# Create trigger - at startup and every day
$trigger1 = New-ScheduledTaskTrigger -AtStartup
$trigger2 = New-ScheduledTaskTrigger -Daily -At "00:00"

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 999 -ExecutionTimeLimit (New-TimeSpan -Days 365)

# Create principal (run whether logged in or not)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger1,$trigger2 -Settings $settings -Principal $principal -Description "24/7 Forex Full Coverage - TrueFX + IB Gateway + OANDA - Paper Trading $100"

Write-Host ""
Write-Host "============================================================"
Write-Host "FOREX 24/7 FULL COVERAGE - SCHEDULED TASK CREATED"
Write-Host "============================================================"
Write-Host "Task Name: $taskName"
Write-Host "Mode: Paper trading with $100 capital"
Write-Host "Data Sources: TrueFX + IB Gateway + OANDA (70+ pairs)"
Write-Host "Triggers: At startup + Daily at midnight"
Write-Host "Auto-restart: Every 1 minute if stopped"
Write-Host ""
Write-Host "To start now: Start-ScheduledTask -TaskName '$taskName'"
Write-Host "To check status: Get-ScheduledTask -TaskName '$taskName'"
Write-Host "To stop: Stop-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$taskName'"
Write-Host "============================================================"
