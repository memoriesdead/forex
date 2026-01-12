# Setup Automatic Live Data Sync
# Runs every 5 minutes to keep live data fresh from Oracle Cloud

Write-Host "Setting up automatic live data sync..." -ForegroundColor Cyan
Write-Host ""

# Create scheduled task action
$action = New-ScheduledTaskAction -Execute "python" -Argument "C:\Users\kevin\forex\scripts\sync_live_data.py" -WorkingDirectory "C:\Users\kevin\forex"

# Create trigger - every 5 minutes
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration ([TimeSpan]::MaxValue)

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register task
try {
    Register-ScheduledTask -TaskName "ForexLiveDataSync" -Action $action -Trigger $trigger -Settings $settings -Description "Sync forex live data from Oracle Cloud every 5 minutes" -Force

    Write-Host "[SUCCESS] Live data sync task created!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Schedule: Every 5 minutes"
    Write-Host "Keeps local live data fresh for paper trading"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  Check status:  Get-ScheduledTask -TaskName 'ForexLiveDataSync'"
    Write-Host "  Run now:       Start-ScheduledTask -TaskName 'ForexLiveDataSync'"
    Write-Host "  Disable:       Disable-ScheduledTask -TaskName 'ForexLiveDataSync'"
    Write-Host "  Delete:        Unregister-ScheduledTask -TaskName 'ForexLiveDataSync' -Confirm:`$false"
    Write-Host ""

    # Run once now to test
    Write-Host "Running sync now to test..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName "ForexLiveDataSync"
    Write-Host "Sync started in background. Check logs/live_sync.log for progress."

} catch {
    Write-Host "[ERROR] Failed to create task: $_" -ForegroundColor Red
    Write-Host "Try running PowerShell as Administrator" -ForegroundColor Yellow
}
