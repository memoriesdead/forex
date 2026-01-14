# HFT Forex ML Trading System

High-frequency forex trading system using 476-feature ML ensemble (Alpha101, Alpha191, Renaissance signals).

## Quick Start (New PC)

```bash
# 1. Clone repo
git clone https://github.com/memoriesdead/forex.git
cd forex

# 2. Get secrets from Oracle Cloud
scp ubuntu@89.168.65.47:/home/ubuntu/projects/forex/.env .
scp ubuntu@89.168.65.47:/home/ubuntu/projects/forex/.env.backup .
scp ubuntu@89.168.65.47:/home/ubuntu/projects/forex/.env_hostinger .
scp ubuntu@89.168.65.47:/home/ubuntu/projects/forex/ssh-key*.key .

# 3. Install dependencies
pip install -r requirements_ml.txt
pip install -r requirements_institutional.txt

# 4. Train models (RTX 5080)
cd training_package
python train_models.py

# 5. Start paper trading
python scripts/start_hft.py paper --symbols EURUSD,GBPUSD
```

## Oracle Cloud Access

```bash
# SSH to Oracle Cloud (need SSH key first)
ssh -i "ssh-key-2026-01-07 (1).key" ubuntu@89.168.65.47

# IB Gateway tunnel (for trading)
ssh -i "ssh-key-2026-01-07 (1).key" -L 4001:localhost:4001 ubuntu@89.168.65.47 -N
```

## Project Structure

```
forex/
├── core/                 # 44 ML modules (HFT engine, Alpha101, signals)
├── scripts/              # Trading bots, training scripts
├── training_package/     # Pre-processed training data
├── models/               # Trained models
├── config/               # Trading session configs
└── libs/                 # Chinese quant formulas
```

## Features

- 476 features per tick (Alpha101, Alpha191, Renaissance, Chinese HFT)
- XGBoost/LightGBM/CatBoost ensemble
- IB Gateway integration via Oracle Cloud
- Walk-forward validation with purged CV

## Important

**Secrets NOT in repo** - get from Oracle Cloud:
- `.env` (API keys)
- `.env.backup`
- `.env_hostinger`
- `ssh-key-2026-01-07 (1).key`
