@echo off
REM Setup Windows Scheduled Task for Live Data Sync
REM Runs every 5 minutes

echo Setting up live data sync task...

schtasks /create /tn "ForexLiveDataSync" /tr "python C:\Users\kevin\forex\scripts\sync_live_data.py" /sc minute /mo 5 /ru SYSTEM /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Live data sync task created!
    echo Task will run every 5 minutes automatically.
    echo.
    echo To check status:  schtasks /query /tn "ForexLiveDataSync"
    echo To run now:       schtasks /run /tn "ForexLiveDataSync"
    echo To disable:       schtasks /change /tn "ForexLiveDataSync" /disable
    echo To delete:        schtasks /delete /tn "ForexLiveDataSync" /f
    echo.
) else (
    echo.
    echo [ERROR] Failed to create task. Run as Administrator.
    echo.
)

pause
