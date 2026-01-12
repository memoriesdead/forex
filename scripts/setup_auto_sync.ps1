# Setup Automatic Historical Data Sync
# Runs every 6 hours to sync from Oracle Cloud

Write-Host "Setting up automatic historical data sync..." -ForegroundColor Cyan
Write-Host ""

# Create scheduled task action
$action = New-ScheduledTaskAction -Execute "python" -Argument "C:\Users\kevin\forex\scripts\sync_historical_data.py" -WorkingDirectory "C:\Users\kevin\forex"

# Create trigger - every 6 hours at :30 minutes (30 min after Oracle downloads)
$trigger = New-ScheduledTaskTrigger -Daily -At "00:30" -DaysInterval 1
$trigger.Repetition = $(New-ScheduledTaskTrigger -Once -At "00:30" -RepetitionInterval (New-TimeSpan -Hours 6) -RepetitionDuration ([TimeSpan]::MaxValue)).Repetition

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register task
try {
    Register-ScheduledTask -TaskName "ForexHistoricalDataSync" -Action $action -Trigger $trigger -Settings $settings -Description "Sync forex historical data from Oracle Cloud every 6 hours" -Force

    Write-Host "[SUCCESS] Auto-sync task created!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Schedule: Every 6 hours at :30 minutes (00:30, 06:30, 12:30, 18:30)"
    Write-Host "This syncs 30 minutes after Oracle Cloud finishes downloading"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  Check status:  Get-ScheduledTask -TaskName 'ForexHistoricalDataSync'"
    Write-Host "  Run now:       Start-ScheduledTask -TaskName 'ForexHistoricalDataSync'"
    Write-Host "  Disable:       Disable-ScheduledTask -TaskName 'ForexHistoricalDataSync'"
    Write-Host "  Delete:        Unregister-ScheduledTask -TaskName 'ForexHistoricalDataSync' -Confirm:`$false"
    Write-Host ""

    # Run once now to test
    Write-Host "Running sync now to test..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName "ForexHistoricalDataSync"
    Write-Host "Sync started in background. Check logs/historical_sync.log for progress."

} catch {
    Write-Host "[ERROR] Failed to create task: $_" -ForegroundColor Red
    Write-Host "Try running PowerShell as Administrator" -ForegroundColor Yellow
}
