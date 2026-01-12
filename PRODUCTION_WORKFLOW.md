# Production Forex Trading Workflow

**Updated:** 2026-01-08 4:20pm PST

## THE ONLY WORKFLOW (DO NOT DEVIATE)

### Step 1: Train Models on Vast.ai H100

```bash
python scripts/train_all_frameworks_h100.py
```

**What it does:**
1. Rents Vast.ai H100/A100 GPU ($1-3/hour)
2. Uploads data from `data/live/` to GPU
3. Trains all frameworks: Qlib, FinRL, V20pyPro, Chinese Quant, GitHub repos
4. Downloads trained models to `models/`
5. Destroys instance automatically

**Duration:** ~25 minutes
**Cost:** ~$1-1.50

**CRITICAL:** If API key invalid, get new one from https://console.vast.ai/account/ and update `.env`

### Step 2: Live Trading with Trained Models

```bash
python start_trading.py --mode paper --session auto --broker ib
```

**What it does:**
1. Connects to Interactive Brokers (paper account)
2. Detects current session (morning/afternoon/evening)
3. Uses trained models from `models/`
4. Executes trades on USD/JPY, AUD/USD (afternoon session)

**Optional:** Use Vast.ai for live inference (more powerful):

```bash
python scripts/vastai_live_inference.py
```

Sends live tick data to Vast.ai GPU for real-time predictions (faster than local).

## Files That Matter

**Training:**
- `scripts/train_all_frameworks_h100.py` - ONLY training script

**Trading:**
- `start_trading.py` - Main entry point
- `scripts/ib_paper_trading_bot.py` - IB integration
- `scripts/session_aware_trading_bot.py` - OANDA integration

**Data:**
- `data/live/` - Live tick data (TrueFX 24/7 capture on Oracle Cloud)
- `models/` - Trained models (downloaded from Vast.ai)

**Config:**
- `.env` - API keys (Vast.ai, IB, OANDA)
- `config/trading_sessions.json` - Session parameters

## What NOT To Do

❌ NEVER create new training scripts
❌ NEVER run training locally
❌ NEVER run ML compute locally
❌ NEVER create "quick test" scripts
❌ NEVER archive - production only

## What To Do

✅ Use existing scripts only
✅ Train on Vast.ai H100
✅ Trade locally with IB using trained models
✅ Keep codebase clean

## Current Status

- ✅ Production workflow documented
- ✅ Rules saved to CLAUDE.md and MCP
- ⚠️ Vast.ai API key INVALID (needs update)
- ✅ IB connection working (DUO423364, $1M balance)
- ✅ Live data capturing 24/7 on Oracle Cloud

## Goal

**$100 → Maximum ROI using H100-trained models on USD/JPY + AUD/USD**

Once Vast.ai key is fixed: Train → Trade → Profit.
