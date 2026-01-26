# HFT Forex ML Trading System

High-frequency forex trading system using ML ensemble (XGBoost/LightGBM/CatBoost) with 73+ features.

## Current Setup (RTX 5080 PC)

**Training:** Local GPU (RTX 5080)
**Trading:** IB Gateway via Oracle Cloud (89.168.65.47)
**Data:** Pre-processed in `training_package/` (EURUSD, GBPUSD, USDJPY)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_ml.txt

# 2. Train models on RTX 5080
cd training_package
python train_models.py

# 3. Connect to IB Gateway (separate terminal)
ssh -i "ssh-key-2026-01-07 (1).key" -L 4001:localhost:4001 ubuntu@89.168.65.47 -N

# 4. Start paper trading
python scripts/start_hft.py paper --symbols EURUSD,GBPUSD
```

## Project Structure

```
forex/
├── core/                 # ML modules (HFT engine, Alpha101, signals)
├── scripts/              # Trading bots, training scripts
├── training_package/     # Pre-processed training data (parquet)
│   ├── EURUSD/          # 129k train, 27k val, 27k test samples
│   ├── GBPUSD/
│   └── USDJPY/
├── models/               # Trained models
│   └── hft_ensemble/    # Production models
├── config/               # Trading session configs
└── libs/                 # Chinese quant formulas
```

## Training Data

| Symbol | Train | Val | Test | Features |
|--------|-------|-----|------|----------|
| EURUSD | 129,804 | 27,815 | 27,816 | 73 |
| GBPUSD | 129,804 | 27,815 | 27,816 | 73 |
| USDJPY | 129,804 | 27,815 | 27,816 | 73 |

## Model Ensemble

- **XGBoost** - GPU-accelerated (tree_method=gpu_hist)
- **LightGBM** - GPU-accelerated
- **CatBoost** - GPU-accelerated
- **Targets:** Direction prediction at 1, 5, 10, 20, 50, 100 tick horizons

## Oracle Cloud (170.9.13.229)

| Service | Port | Purpose |
|---------|------|---------|
| IB Gateway API | 4001 | Trading execution |
| IB Gateway VNC | 5900 | Manual login (password: ibgateway) |

```bash
# SSH access
ssh -i "ssh-key-oracle.key" ubuntu@170.9.13.229

# IB Gateway tunnel (run in separate terminal)
ssh -i "ssh-key-oracle.key" -L 4001:localhost:4001 ubuntu@170.9.13.229 -N
```

## Files

- `ssh-key-oracle.key` - SSH key for Oracle Cloud
- `.env` - Environment variables (API keys, config)
