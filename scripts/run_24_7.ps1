# 24/7 FOREX TRADING BOT - Auto-restart on failure
# Run: powershell -ExecutionPolicy Bypass -File scripts\run_24_7.ps1

$ErrorActionPreference = "Continue"
$scriptPath = "C:\Users\kevin\forex\forex"
$logFile = "$scriptPath\logs\trading_24_7.log"
$symbols = "EURUSD,GBPUSD,USDJPY,EURJPY,GBPJPY,AUDUSD,USDCAD,USDCHF,NZDUSD,EURGBP,EURAUD,GBPAUD,AUDNZD,AUDJPY,CADJPY"
$capital = 200

# Create logs directory
New-Item -ItemType Directory -Force -Path "$scriptPath\logs" | Out-Null

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Tee-Object -FilePath $logFile -Append
}

function Start-TradingBot {
    Write-Log "Starting trading bot..."

    # Activate venv and run
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c cd /d $scriptPath && venv\Scripts\activate && python scripts/hft_trading_bot.py --mode paper --symbols $symbols --capital $capital --multi-source 2>&1" -NoNewWindow -PassThru -RedirectStandardOutput "$scriptPath\logs\bot_output.log" -RedirectStandardError "$scriptPath\logs\bot_error.log"

    return $process
}

function Check-IBGateway {
    $container = docker ps --filter "name=ibgateway" --format "{{.Status}}" 2>$null
    if ($container -match "Up") {
        return $true
    }
    Write-Log "IB Gateway not running, starting..."
    docker start ibgateway 2>$null
    Start-Sleep -Seconds 30
    return $false
}

function Check-MCP {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8082/health" -TimeoutSec 5
        return $true
    } catch {
        Write-Log "MCP server not running, starting..."
        Start-Process -FilePath "python" -ArgumentList "$scriptPath\mcp_servers\llm_live_tuning_mcp.py --port 8082" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        return $false
    }
}

# Main loop
Write-Log "=========================================="
Write-Log "24/7 FOREX TRADING SYSTEM STARTING"
Write-Log "Capital: $$capital | Symbols: $symbols"
Write-Log "=========================================="

$restartCount = 0
$maxRestarts = 100  # Max restarts before pause

while ($true) {
    # Check dependencies
    Check-IBGateway | Out-Null
    Check-MCP | Out-Null

    # Start bot
    $botProcess = Start-TradingBot
    Write-Log "Bot started with PID: $($botProcess.Id)"

    # Monitor bot
    while (-not $botProcess.HasExited) {
        Start-Sleep -Seconds 30

        # Check if still profitable (optional health check)
        $lastLine = Get-Content "$scriptPath\logs\bot_output.log" -Tail 1 -ErrorAction SilentlyContinue
        if ($lastLine) {
            # Log periodic status
            if ((Get-Date).Minute % 5 -eq 0) {
                Write-Log "Bot running... Last: $lastLine"
            }
        }
    }

    # Bot exited
    $exitCode = $botProcess.ExitCode
    Write-Log "Bot exited with code: $exitCode"
    $restartCount++

    if ($restartCount -ge $maxRestarts) {
        Write-Log "Max restarts reached ($maxRestarts). Pausing 1 hour..."
        Start-Sleep -Seconds 3600
        $restartCount = 0
    }

    # Wait before restart
    Write-Log "Restarting in 10 seconds... (restart #$restartCount)"
    Start-Sleep -Seconds 10
}
