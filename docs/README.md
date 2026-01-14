# Forex Trading System - Documentation

**All documentation organized in docs/ folder**

---

## Quick Start

**Start trading:**
```bash
python start_trading.py --mode paper --session auto --broker ib
```

**Backtest strategy:**
```bash
python backtest.py --strategy ml_ensemble --start 2026-01-01 --end 2026-01-08
```

---

## Documentation Structure

### Setup Guides (`docs/setup/`)
- System installation
- Broker configuration (IB, OANDA)
- Data pipeline setup
- ML framework installation

### Trading Guides (`docs/trading/`)
- Trading sessions (6:30am, 9pm)
- Strategy documentation
- Risk management
- Performance tracking

### ML Documentation (`docs/ml/`)
- Time-Series-Library (47 models)
- Qlib integration
- FinRL DRL agents
- Chinese quant frameworks
- Renaissance signals (50+)

### Infrastructure (`docs/infrastructure/`)
- Oracle Cloud setup
- MCP server configuration
- Data sync procedures
- System architecture

---

## Key Files

### Core Entry Points
- `start_trading.py` - Main trading script
- `backtest.py` - Backtesting engine
- `CLAUDE.md` - Project rules and guidelines

### Core Modules
- `core/ml_integration.py` - ML ensemble framework
- `core/renaissance_signals.py` - 50+ weak signals
- `config/trading_sessions.json` - Session parameters

### Scripts
- `scripts/ib_paper_trading_bot.py` - Interactive Brokers
- `scripts/session_aware_trading_bot.py` - OANDA
- `scripts/sync_live_data_v2.py` - Data sync

---

## Trading Sessions

### Morning (6:30am-9am PST)
- **Pairs:** EUR/USD, GBP/USD, USD/JPY
- **Strategy:** Aggressive breakout/trend
- **Capital:** 70%

### Evening (9pm-11:59pm PST)
- **Pairs:** USD/JPY, AUD/USD
- **Strategy:** Conservative range/reversion
- **Capital:** 30%

---

## ML Frameworks

### Time-Series-Library (47 Models)
Autoformer, Informer, TimesNet, DLinear, FEDformer, ETSformer, etc.

### Qlib (Microsoft)
Quant research platform, factor analysis, workflow management

### FinRL (Deep RL)
PPO, A2C, DDPG, TD3, SAC agents for trading

### Chinese Quant
QUANTAXIS, vnpy, Tushare (install: `pip install quantaxis vnpy`)

### Renaissance Signals
50+ weak signals (65-70% accuracy combined)

---

## Installation

**Install ML dependencies:**
```bash
pip install -r requirements_ml.txt
```

**Install Chinese quant:**
```bash
pip install quantaxis vnpy
```

---

## Performance Targets

### Validation Criteria
- ✓ Win rate > 55%
- ✓ Sharpe ratio > 1.5
- ✓ Max drawdown < 5%
- ✓ At least 30 trades

### Expected Performance
- **Baseline:** 55-60% win rate, 1.5-2.0 Sharpe
- **ML Ensemble:** 60-65% win rate, 2.0-2.5 Sharpe
- **Renaissance:** 65-70% win rate, 2.5-3.0 Sharpe

---

## Data Pipeline

**Live capture (24/7):**
- TrueFX API: 10 pairs, 1-second ticks
- Oracle Cloud: Systemd service running
- Local sync: On demand

**Sync data:**
```bash
scp -i "ssh-key-2026-01-07 (1).key" -r ubuntu@89.168.65.47:/home/ubuntu/projects/forex/data/live/2026-01-08/*.csv data/live/2026-01-08/
```

---

## Configuration

### Trading Sessions
`config/trading_sessions.json` - Session parameters, pairs, risk limits

### Broker Settings
`.env` - IB/OANDA credentials (NEVER COMMIT)

### ML Models
`models/` - Trained models (HFT, Time-Series, FinRL)

---

## Support

**Main project guide:** `CLAUDE.md`
**This documentation:** `docs/README.md`
**Quick commands:** See scripts above

**For issues:** Check logs in `logs/trading/`
