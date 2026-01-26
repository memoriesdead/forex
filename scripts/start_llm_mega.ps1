<#
.SYNOPSIS
    FOREX LLM MEGA TRADING SYSTEM - FULL INTEGRATION
    DeepSeek-R1 + 806 Features + Multi-Agent Reasoning

.DESCRIPTION
    Starts the complete trading system with:
    - Ollama service (DeepSeek-R1-8B)
    - ML Ensemble (XGBoost + LightGBM + CatBoost)
    - Chinese Quant Online Learning
    - Multi-Agent LLM Reasoning (Bull/Bear debate)
    - Full Coverage Multi-Source Data

.PARAMETER Capital
    Starting capital in USD (default: 100)

.PARAMETER Symbols
    Comma-separated list of symbols (default: majors)

.PARAMETER Mode
    Trading mode: paper, live, monitor (default: paper)

.PARAMETER LLMMode
    LLM integration mode: advisory, validation, autonomous (default: validation)

.EXAMPLE
    .\start_llm_mega.ps1 -Capital 1000 -Mode paper
    .\start_llm_mega.ps1 -Capital 10000 -Mode live -LLMMode autonomous
#>

param(
    [int]$Capital = 100,
    [string]$Symbols = "EURUSD,GBPUSD,USDJPY",
    [string]$Mode = "paper",
    [string]$LLMMode = "validation"
)

$ErrorActionPreference = "Stop"

Write-Host @"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FOREX LLM MEGA TRADING SYSTEM                             ║
║                         FULL SEND MODE ACTIVATED                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ML Ensemble:     XGBoost + LightGBM + CatBoost (63% base accuracy)          ║
║  LLM Reasoner:    DeepSeek-R1 Multi-Agent (Bull/Bear/Risk debate)            ║
║  Features:        806+ alpha factors (Alpha101 + Alpha191 + Renaissance)     ║
║  Online Learning: Chinese quant style (幻方/九坤/明汯)                        ║
║  Target:          63% → 75%+ win rate with reasoning                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Capital:  `$$Capital"
Write-Host "  Symbols:  $Symbols"
Write-Host "  Mode:     $Mode"
Write-Host "  LLM Mode: $LLMMode"
Write-Host ""

# Check Ollama
Write-Host "[1/4] Checking Ollama service..." -ForegroundColor Green
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

$ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaRunning) {
    Write-Host "  Starting Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    Write-Host "  Ollama started" -ForegroundColor Green
} else {
    Write-Host "  Ollama already running" -ForegroundColor Green
}

# Check DeepSeek model
Write-Host "[2/4] Checking DeepSeek-R1 model..." -ForegroundColor Green
$models = & $ollamaPath list 2>$null
if ($models -match "deepseek-r1:8b") {
    Write-Host "  DeepSeek-R1:8B ready (5.2GB)" -ForegroundColor Green
} else {
    Write-Host "  Pulling DeepSeek-R1:8B (this may take a while)..." -ForegroundColor Yellow
    & $ollamaPath pull deepseek-r1:8b
}

# Check GPU
Write-Host "[3/4] Checking GPU status..." -ForegroundColor Green
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>$null
    if ($gpu) {
        Write-Host "  GPU: $gpu" -ForegroundColor Green
    }
} catch {
    Write-Host "  GPU check skipped" -ForegroundColor Yellow
}

# Start trading bot
Write-Host "[4/4] Starting HFT Trading Bot..." -ForegroundColor Green
Write-Host ""

Set-Location "C:\Users\kevin\forex\forex"

# Activate venv and run
$env:PYTHONPATH = "C:\Users\kevin\forex\forex"

Write-Host @"
╔══════════════════════════════════════════════════════════════════════════════╗
║                           TRADING SYSTEM STARTING                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Watch for:                                                                  ║
║    [LLM] DeepSeek-R1 Multi-Agent Reasoner enabled                           ║
║    [LLM] Mode: VALIDATION (LLM can veto ML signals)                         ║
║    [SIGNAL] dir=1/−1, conf=0.XX, LLM:OK/VETO                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

& "C:\Users\kevin\forex\forex\venv\Scripts\python.exe" `
    "C:\Users\kevin\forex\forex\scripts\hft_trading_bot.py" `
    --mode $Mode `
    --symbols $Symbols `
    --capital $Capital `
    --full-coverage `
    --multi-source

Write-Host ""
Write-Host "Trading session ended." -ForegroundColor Yellow
