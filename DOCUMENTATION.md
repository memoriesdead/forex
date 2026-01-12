# Forex Trading System - Documentation Index

**Last Updated:** 2026-01-08 3:30pm PST

---

## Quick Start

```bash
# Start trading
python start_trading.py --mode paper --session auto --broker ib

# Run backtest
python backtest.py --strategy ml_ensemble --start 2026-01-01 --end 2026-01-08

# View documentation
cat docs/README.md
```

---

## Documentation Locations

### Active Documentation
**Location:** `docs/` folder

**Main files:**
- `docs/README.md` - Master documentation index
- `CLAUDE.md` - Project rules and ML-first approach
- `README.md` - This file

### Archived Documentation
**Location:** `archive/docs-2026-01-08/`

**Contains:**
- All historical documentation from organization session
- Setup guides, trading guides, ML guides
- Reference material (29 files)

**Access:** See `archive/docs-2026-01-08/RESTORE_INSTRUCTIONS.md`

### Configuration Storage
**Location:** MCP Server (Oracle Cloud)

**Retrieve:**
```python
mcp__memory-keeper__context_get(key="ml_integration_complete")
mcp__memory-keeper__context_get(key="renaissance_methodology")
mcp__memory-keeper__context_get(key="chinese_quant_frameworks")
```

---

## Core Components

### Entry Points
- `start_trading.py` - Main trading script
- `backtest.py` - Backtesting engine

### ML Framework
- `core/ml_integration.py` - 47 models ensemble
- `core/renaissance_signals.py` - 50+ weak signals

### Configuration
- `config/trading_sessions.json` - Session parameters
- `.env` - API credentials (never commit)

### Trading Sessions
- **Morning:** 6:30am-9am PST (EUR/USD, GBP/USD, USD/JPY)
- **Evening:** 9pm-11:59pm PST (USD/JPY, AUD/USD)

---

## ML Frameworks Available

1. **Time-Series-Library** - 47 forecasting models
2. **Qlib** - Microsoft quant research
3. **FinRL** - Deep RL agents (PPO, A2C, DDPG, TD3, SAC)
4. **Chinese Quant** - QUANTAXIS, vnpy
5. **Renaissance** - 50+ weak signals (65-70% accuracy)

**Install:**
```bash
pip install -r requirements_ml.txt
```

---

## Data Pipeline

**Live capture (24/7):**
- TrueFX API: 10 pairs, 1-second ticks
- Oracle Cloud: Systemd service running
- Local sync: On demand

**Status:**
```bash
ssh -i "ssh-key-2026-01-07 (1).key" ubuntu@89.168.65.47 "systemctl status forex-live-capture"
```

---

## Performance Targets

**Validation criteria:**
- Win rate > 55%
- Sharpe ratio > 1.5
- Max drawdown < 5%
- Minimum 30 trades

**Expected performance:**
- Baseline: 55-60% win rate
- ML Ensemble: 60-65% win rate
- Renaissance: 65-70% win rate

---

## Project Rules (ML-First)

From `CLAUDE.md`:

1. **Always backtest first** - Never trade untested strategies
2. **ML predictions > manual signals** - Trust the data
3. **Ensemble > single model** - Diversify predictions
4. **Paper trade 2 weeks** - Validate in real conditions
5. **Risk management > returns** - Preserve capital

---

## Folder Structure

```
forex/
├── CLAUDE.md                   ← Project rules
├── DOCUMENTATION.md            ← This file
├── start_trading.py            ← Main entry
├── backtest.py                 ← Backtesting
├── requirements_ml.txt         ← ML dependencies
│
├── docs/                       ← Active documentation
│   ├── README.md
│   ├── setup/
│   ├── trading/
│   ├── ml/
│   └── infrastructure/
│
├── archive/                    ← Historical docs
│   └── docs-2026-01-08/       ← Today's session (29 files)
│
├── core/                       ← Core modules
│   ├── ml_integration.py      ← ML ensemble
│   └── renaissance_signals.py ← 50+ weak signals
│
├── config/                     ← Configuration
├── scripts/                    ← Utility scripts
├── data/                       ← Data storage
├── models/                     ← Trained models
└── logs/                       ← System logs
```

---

## Support

**Questions about:**
- **Setup:** See `docs/setup/`
- **Trading:** See `docs/trading/`
- **ML:** See `docs/ml/`
- **Infrastructure:** See `docs/infrastructure/`

**Archived reference:** See `archive/docs-2026-01-08/`

**Live help:** Ask Claude (this conversation context preserved)

---

**Last session:** 2026-01-08 (Organization + ML implementation)
**Status:** Production-ready
**Next:** Test at 9pm PST
