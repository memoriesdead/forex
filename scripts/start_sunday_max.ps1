# MAXIMUM AGGRESSIVE SUNDAY AUTO-START
# Pure algorithmic - HANDS OFF trading
#
# Configuration:
# - 77.5% verified win rate
# - 55% Full Kelly
# - 5 second cooldown
# - 10,000+ trades/day capacity
# - Continuous Chinese quant learning
# - 51 TRAINED PAIRS (all majors, crosses, exotics)

param(
    [int]$Capital = 200,
    [string]$Mode = "paper",  # Change to "live" for real money
    [switch]$FullCoverage = $true  # Trade ALL 51 trained pairs
)

$ForexDir = "C:\Users\kevin\forex\forex"
Set-Location $ForexDir

Write-Host @"

================================================================================
             MAXIMUM AGGRESSIVE ALGORITHMIC TRADING
                   PURE HANDS-OFF MODE
================================================================================
  Capital:     $Capital
  Mode:        $Mode
  Symbols:     51 TRAINED PAIRS (--full-coverage)
  Kelly:       55% (FULL)
  Cooldown:    5 seconds
  Win Rate:    77.5% verified
================================================================================

"@ -ForegroundColor Green

# Check if IB Gateway is running
$docker = docker ps --filter "name=ibgateway" --format "{{.Names}}" 2>$null
if (-not $docker) {
    Write-Host "[ERROR] IB Gateway not running. Starting..." -ForegroundColor Red
    docker start ibgateway
    Start-Sleep -Seconds 10
}

# Wait for market open if Sunday before 5 PM ET
$et = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), 'Eastern Standard Time')
$day = $et.DayOfWeek
$hour = $et.Hour

Write-Host "[INFO] Current ET time: $($et.ToString('dddd HH:mm:ss'))" -ForegroundColor Cyan

if ($day -eq 'Sunday' -and $hour -lt 17) {
    $marketOpen = Get-Date -Year $et.Year -Month $et.Month -Day $et.Day -Hour 17 -Minute 0 -Second 0
    $waitMinutes = ($marketOpen - $et).TotalMinutes
    Write-Host "[WAIT] Market opens Sunday 5 PM ET. Waiting $([math]::Round($waitMinutes)) minutes..." -ForegroundColor Yellow

    # Wait with countdown
    while ((Get-Date) -lt $marketOpen) {
        $remaining = ($marketOpen - (Get-Date)).TotalMinutes
        Write-Host "`r[WAIT] Market opens in $([math]::Round($remaining)) minutes...    " -NoNewline -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
    Write-Host ""
}

# Check if market is closed (Saturday)
if ($day -eq 'Saturday') {
    Write-Host "[ERROR] Market is CLOSED on Saturday. Come back Sunday 5 PM ET." -ForegroundColor Red
    exit 1
}

# Start MCP Live Tuning Server in background
Write-Host "[START] MCP Live Tuning Server on port 8082..." -ForegroundColor Cyan
$mcpJob = Start-Job -ScriptBlock {
    Set-Location $using:ForexDir
    & "$using:ForexDir\venv\Scripts\python.exe" "$using:ForexDir\mcp_servers\llm_live_tuning_mcp.py" --port 8082
}
Start-Sleep -Seconds 3

# Verify MCP server
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8082/health" -TimeoutSec 5
    Write-Host "[OK] MCP Server healthy: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "[WARN] MCP Server not responding, continuing anyway..." -ForegroundColor Yellow
}

# Build command - trades ALL 51 trained pairs
$cmd = @(
    "$ForexDir\venv\Scripts\python.exe",
    "$ForexDir\scripts\hft_trading_bot.py",
    "--mode", $Mode,
    "--capital", $Capital,
    "--full-coverage"
)

Write-Host @"

================================================================================
                    STARTING TRADING BOT
================================================================================
  Command: $($cmd -join " ")

  SETTINGS (MAXIMUM AGGRESSIVE):
  - Kelly Fraction: 55% (FULL KELLY)
  - Max Daily Trades: 10,000
  - Trade Cooldown: 5 seconds
  - Take Profit: 10 pips OR 0.2%
  - Max Hold: 15 minutes
  - Base Accuracy: 77.5%

  HANDS OFF - PURE ALGORITHMIC TRADING
================================================================================

"@ -ForegroundColor Green

# Run the trading bot
& $cmd[0] $cmd[1..($cmd.Length-1)]

# Cleanup MCP job on exit
Stop-Job $mcpJob -ErrorAction SilentlyContinue
Remove-Job $mcpJob -ErrorAction SilentlyContinue
