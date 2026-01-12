@echo off
REM Oracle Cloud SSH Tunnel for MCP Servers
REM Run this at startup or before using Claude Code

echo Starting SSH tunnel to Oracle Cloud...
echo Ports: 37777 (claude-mem), 3847 (memory-keeper)

ssh -L 37777:localhost:37777 -L 3847:localhost:3847 -i "C:\Users\kevin\forex\ssh-key-2026-01-07 (1).key" -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -N ubuntu@89.168.65.47

echo Tunnel closed.
pause
