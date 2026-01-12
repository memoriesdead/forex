@echo off
REM Setup automatic historical data sync from Oracle Cloud to local
REM Runs every 6 hours to match Oracle Cloud download schedule

echo Setting up automatic historical data sync...
echo.

REM Create scheduled task to run every 6 hours
schtasks /create /tn "ForexHistoricalDataSync" /tr "python C:\Users\kevin\forex\scripts\sync_historical_data.py" /sc hourly /mo 6 /st 00:30 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Auto-sync task created!
    echo.
    echo Schedule: Every 6 hours at :30 minutes (00:30, 06:30, 12:30, 18:30)
    echo This syncs 30 minutes after Oracle Cloud finishes downloading
    echo.
    echo Commands:
    echo   Check status: schtasks /query /tn "ForexHistoricalDataSync"
    echo   Run now:      schtasks /run /tn "ForexHistoricalDataSync"
    echo   Disable:      schtasks /change /tn "ForexHistoricalDataSync" /disable
    echo   Delete:       schtasks /delete /tn "ForexHistoricalDataSync" /f
    echo.
) else (
    echo.
    echo [ERROR] Failed to create task. Run as Administrator.
    echo.
)

pause
