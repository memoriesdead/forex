@echo off
REM SSH Tunnel for IB Gateway VNC Access
REM Forwards localhost:5900 to Oracle Cloud ibgateway VNC
REM VNC Password: ibgateway

echo ============================================
echo IB Gateway VNC Tunnel
echo ============================================
echo.
echo Starting SSH tunnel to Oracle Cloud...
echo VNC will be available at: localhost:5900
echo VNC Password: ibgateway
echo.
echo After connecting via VNC:
echo   1. You should see the IB Gateway login screen
echo   2. Enter your credentials if prompted
echo   3. Complete any 2FA verification
echo   4. Once logged in, you can close VNC
echo.
echo Press Ctrl+C to stop the tunnel
echo ============================================
echo.

ssh -i "C:\Users\kevin\forex\ssh-key-2026-01-07 (1).key" -o StrictHostKeyChecking=no -L 5900:localhost:5900 ubuntu@89.168.65.47 -N
