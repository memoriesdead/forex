@echo off
REM Start Real-Time Live Data Stream (Oracle Cloud -> Local, PST timestamps)
REM Updates EVERY SECOND, NO DELAYS

echo ============================================================
echo STARTING REAL-TIME LIVE DATA STREAM
echo ============================================================
echo.
echo Frequency: Every 1 SECOND
echo Timezone: PST (America/Los_Angeles)
echo Source: Oracle Cloud (89.168.65.47)
echo Destination: data/truefx_live/
echo.
echo Press Ctrl+C to stop
echo ============================================================
echo.

cd /d C:\Users\kevin\forex
python scripts\live_stream_sync.py

pause
