#Requires -Version 5.1
<#
.SYNOPSIS
    Full Coverage Forex Trading Bot - Maximum Data Sources
.DESCRIPTION
    Starts the HFT trading bot with data from ALL sources:
    - TrueFX (10-15 majors, free)
    - IB Gateway (70+ pairs)
    - OANDA (28+ pairs)

    With $100 capital and Chinese Quant online learning.
.PARAMETER Capital
    Starting capital in USD (default: 100)
.PARAMETER Mode
    Trading mode: paper or live (default: paper)
.PARAMETER Symbols
    Comma-separated symbols or 'ALL' for all (default: ALL)
.EXAMPLE
    .\start_full_coverage.ps1
    .\start_full_coverage.ps1 -Capital 500 -Mode live
#>

param(
    [Parameter()]
    [double]$Capital = 100.0,

    [Parameter()]
    [ValidateSet('paper', 'live')]
    [string]$Mode = 'paper',

    [Parameter()]
    [string]$Symbols = 'ALL'
)

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "    FULL COVERAGE FOREX TRADING - `$$Capital CAPITAL" -ForegroundColor Cyan
Write-Host "    Maximum Coverage Across ALL Data Sources" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "DATA SOURCES:" -ForegroundColor Yellow
Write-Host "  1. TrueFX       - Free tick streaming (10-15 majors)" -ForegroundColor White
Write-Host "  2. IB Gateway   - Full market data (70+ pairs)" -ForegroundColor White
Write-Host "  3. OANDA        - Streaming quotes (28+ pairs)" -ForegroundColor White
Write-Host ""

Write-Host "FEATURES:" -ForegroundColor Yellow
Write-Host "  - Best quote selection (lowest spread from all sources)" -ForegroundColor White
Write-Host "  - Automatic failover between sources" -ForegroundColor White
Write-Host "  - Chinese Quant online learning (575+ features)" -ForegroundColor White
Write-Host "  - 51 forex pairs with trained ML models" -ForegroundColor White
Write-Host "  - Kelly Criterion position sizing (50% fraction)" -ForegroundColor White
Write-Host "  - 50:1 leverage, 20% max drawdown" -ForegroundColor White
Write-Host ""

# Change to project directory
Set-Location "C:\Users\kevin\forex\forex"

# Check IB Gateway
Write-Host "Checking IB Gateway..." -ForegroundColor Yellow
$ibGateway = docker ps --filter "name=ibgateway" --format "{{.Names}}" 2>$null

if (-not $ibGateway) {
    Write-Host "[WARNING] IB Gateway not running. Starting..." -ForegroundColor Red
    docker start ibgateway 2>$null
    Start-Sleep -Seconds 5
    $ibGateway = docker ps --filter "name=ibgateway" --format "{{.Names}}" 2>$null
    if ($ibGateway) {
        Write-Host "[OK] IB Gateway started" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Could not start IB Gateway. Continuing with TrueFX only." -ForegroundColor Yellow
    }
} else {
    Write-Host "[OK] IB Gateway is running" -ForegroundColor Green
}

# Check virtual environment
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Starting FULL COVERAGE trading..." -ForegroundColor Cyan
Write-Host "  Mode: $Mode" -ForegroundColor White
Write-Host "  Capital: `$$Capital" -ForegroundColor White
Write-Host "  Symbols: $Symbols" -ForegroundColor White
Write-Host "  Online Learning: Enabled" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Build arguments
$args = @(
    "scripts/hft_trading_bot.py",
    "--mode", $Mode,
    "--capital", $Capital,
    "--multi-source",
    "--online-learning"
)

if ($Symbols -eq 'ALL') {
    $args += "--full-coverage"
} else {
    $args += "--symbols"
    $args += $Symbols
}

# Run trading bot
try {
    python @args
} catch {
    Write-Host ""
    Write-Host "[ERROR] Trading bot crashed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Trading stopped." -ForegroundColor Yellow
