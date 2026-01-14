# HFT Forex ML Trading System

High-frequency forex trading system using 476-feature ML ensemble (Alpha101, Alpha191, Renaissance signals).

## Quick Start (New PC)

```bash
# 1. Clone repo
git clone https://github.com/memoriesdead/forex.git
cd forex

# 2. Get secrets from Oracle Cloud (need SSH key from someone with access first)
scp -i "ssh-key-2026-01-07 (1).key" ubuntu@89.168.65.47:/home/ubuntu/projects/forex/.env .
scp -i "ssh-key-2026-01-07 (1).key" ubuntu@89.168.65.47:/home/ubuntu/projects/forex/.env.backup .
scp -i "ssh-key-2026-01-07 (1).key" ubuntu@89.168.65.47:/home/ubuntu/projects/forex/.env_hostinger .
scp -i "ssh-key-2026-01-07 (1).key" ubuntu@89.168.65.47:/home/ubuntu/projects/forex/ssh-key*.key .

# 3. Install dependencies
pip install -r requirements_ml.txt
pip install -r requirements_institutional.txt

# 4. Train models (RTX 5080)
cd training_package
python train_models.py

# 5. Start paper trading
python scripts/start_hft.py paper --symbols EURUSD,GBPUSD
```

## MCP Server Setup (Oracle Cloud)

The project uses MCP servers on Oracle Cloud for memory-keeper and claude-mem.

### 1. Setup Claude Config

Copy config files to your user directory:
```powershell
# Windows
cp config/claude_settings.json ~/.claude/settings.json
cp config/claude_hooks.json ~/.claude/hooks.json
mkdir ~/.claude-mem
cp -r config/claude-mem/* ~/.claude-mem/
```

### 2. Start Oracle Cloud Tunnel

Before using Claude Code, start the SSH tunnel:
```powershell
powershell -File scripts/start_oracle_tunnel.ps1
```

This forwards:
- `localhost:37777` → claude-mem (auto-capture)
- `localhost:3847` → memory-keeper (manual saves)
- `localhost:4001` → IB Gateway API
- `localhost:5900` → IB Gateway VNC

### 3. MCP Server Config

Add to Claude Desktop config (`%APPDATA%/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "memory-keeper": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-proxy", "http://localhost:3847"]
    }
  }
}
```

## Oracle Cloud Access

```bash
# SSH to Oracle Cloud
ssh -i "ssh-key-2026-01-07 (1).key" ubuntu@89.168.65.47

# Full tunnel (all services)
ssh -i "ssh-key-2026-01-07 (1).key" -L 37777:localhost:37777 -L 3847:localhost:3847 -L 4001:localhost:4001 -L 5900:localhost:5900 ubuntu@89.168.65.47 -N
```

## Project Structure

```
forex/
├── core/                 # 44 ML modules (HFT engine, Alpha101, signals)
├── scripts/              # Trading bots, training scripts
├── training_package/     # Pre-processed training data
├── models/               # Trained models
├── config/               # Trading sessions + MCP configs
├── libs/                 # Chinese quant formulas
└── Time-Series-Library/  # SOTA time series models
```

## Features

- 476 features per tick (Alpha101, Alpha191, Renaissance, Chinese HFT)
- XGBoost/LightGBM/CatBoost ensemble
- IB Gateway integration via Oracle Cloud
- Walk-forward validation with purged CV
- MCP memory-keeper for context persistence

## Important

**Secrets NOT in repo** - get from Oracle Cloud:
- `.env` (API keys)
- `.env.backup`
- `.env_hostinger`
- `ssh-key-2026-01-07 (1).key`

**Raw data NOT in repo** - get from Oracle Cloud:
- `data/` (14GB market data)
- `data_cleaned/` (processed data)
