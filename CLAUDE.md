# Forex Trading ML Project

```
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT: FOREX                                                  ║
║  Path: C:\Users\kevin\forex                                      ║
║  Oracle Cloud: 89.168.65.47 (shared with 5 other projects)       ║
║  ML TRAINING: Vast.ai H100 GPU ONLY (NEVER LOCAL)                ║
╚══════════════════════════════════════════════════════════════════╝
```

## CRITICAL: NO LOCAL TRAINING (COSTS TIME & MONEY)

**Local machine CANNOT handle ML training. ONLY use for:**
1. Code editing
2. IB trading execution (lightweight)
3. Light data syncing

**ML Training/Compute = Vast.ai H100 ONLY:**
- Use `scripts/train_all_frameworks_h100.py`
- Rent H100/A100 GPU on Vast.ai
- Upload data → Train on GPU → Download models
- NEVER create new training scripts locally
- NEVER run training locally

**Violation wastes user's time and money they cannot afford to lose.**

## IBKR Gateway (Oracle Cloud)

**Status:** Running on Oracle Cloud (89.168.65.47)

| Setting | Value |
|---------|-------|
| Account | DUO423364 (paper) |
| API Port | 4001 |
| VNC Port | 5900 |
| VNC Password | ibgateway |
| Container | ibgateway |

**Connect locally via SSH tunnel:**
```powershell
ssh -i "C:\Users\kevin\forex\ssh-key-2026-01-07 (1).key" -L 4001:localhost:4001 ubuntu@89.168.65.47 -N
```
Then connect your bot to `localhost:4001`.

**Check status:**
```bash
ssh ubuntu@89.168.65.47 "docker logs ibgateway | tail -10"
```

---

## HFT System Architecture (2026-01-10)

**Renaissance-Level HFT: 100+ features, tick-level execution, ML ensemble**

### Master Controller
```bash
python scripts/start_hft.py status    # Check system readiness
python scripts/start_hft.py prepare   # Package training data
python scripts/start_hft.py paper     # Start paper trading
python scripts/start_hft.py live      # Start live trading (REAL MONEY)
```

### HFT Core Modules (`core/`)

| Module | File | Purpose |
|--------|------|---------|
| **Order Book L3** | `order_book_l3.py` | L2/L3 reconstruction, imbalance, microprice |
| **Queue Position** | `queue_position.py` | 3 models: RiskAverse, Probabilistic, L3FIFO |
| **Fill Probability** | `fill_probability.py` | Poisson fills, market impact, adverse selection |
| **Tick Backtest** | `tick_backtest.py` | Tick-by-tick simulation engine |
| **Order Flow** | `order_flow_features.py` | OFI, OEI, VPIN, Hawkes intensity |
| **Latency Sim** | `latency_simulator.py` | Feed/order/fill latency modeling |
| **Microstructure** | `microstructure_vol.py` | TSRV, noise filtering, bid-ask bounce |
| **Data Loader** | `hft_data_loader.py` | TrueFX historical + live streaming |
| **Feature Engine** | `hft_feature_engine.py` | 100+ unified features per tick |

### HFT Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `start_hft.py` | Master controller for entire system |
| `prepare_hft_training.py` | Package data for Vast.ai H100 |
| `hft_trading_bot.py` | Production trading bot |

### HFT Workflow (In Order)

**1. Prepare Training Data:**
```bash
python scripts/start_hft.py prepare --symbols EURUSD,GBPUSD,USDJPY --days 30
```

**2. Train on Vast.ai H100 (~$3-5):**
```bash
# Rent H100 on vast.ai (~$2.50/hr)
scp -r training_package/* root@<VAST_IP>:/workspace/
ssh root@<VAST_IP>
./run_training.sh
scp root@<VAST_IP>:/workspace/hft_models.tar.gz .
tar -xzvf hft_models.tar.gz -C models/hft_ensemble/
```

**3. Paper Trade (24hr minimum):**
```bash
python scripts/start_hft.py paper --symbols EURUSD,GBPUSD
```

**4. Go Live:**
```bash
python scripts/start_hft.py live --symbols EURUSD
```

### Feature Categories (100+ Total)

| Category | Count | Source |
|----------|-------|--------|
| Alpha101 | 62 | `alpha101_complete.py` |
| Renaissance | 50 | `renaissance_signals.py` |
| Order Flow | 10 | OFI, OEI, VPIN, Hawkes |
| Microstructure | 8 | TSRV, noise, SNR |
| Technical | 20 | Fast indicators |
| Cross-Asset | 5 | DXY, VIX correlations |

### Model Directory Structure
```
models/
├── hft_ensemble/           # Trained HFT models
│   ├── EURUSD_target_direction_10_models.pkl
│   ├── GBPUSD_target_direction_10_models.pkl
│   └── ...
├── institutional/          # Institutional models
└── gold_standard/          # Legacy models
```

### Risk Management (Built-in)

- **Kelly Criterion**: Fractional Kelly (25%) position sizing
- **Max Drawdown**: 5% hard limit
- **Daily Trade Limit**: 100 trades/day
- **Position Limit**: 2% of account per trade

---

## Gold Standard ML Architecture (2026-01-10)

**Audited against GitHub (35k+ stars) + Gitee Chinese Quant + Renaissance Methods**
**Target: 70%+ accuracy using Vast.ai H100**

### Core Modules (Verified)

| Module | File | Purpose |
|--------|------|---------|
| **HMM Regime** | `core/institutional_predictor.py` | Renaissance 3-state detection |
| **Kalman Filter** | `core/institutional_predictor.py` | Goldman dynamic mean |
| **Alpha101** | `core/alpha101_complete.py` | 62 WorldQuant alphas |
| **Renaissance** | `core/renaissance_signals.py` | 50+ weak signals |
| **Cross-Asset** | `core/cross_asset_signals.py` | DXY, VIX, SPX, Gold, Oil |
| **Walk-Forward** | `core/walk_forward.py` | Purged CV, CPCV |
| **iTransformer** | `core/gold_standard_models.py` | ICLR 2024 time series |
| **TimeXer** | `core/gold_standard_models.py` | NeurIPS 2024 exogenous |
| **Attention Factor** | `core/gold_standard_models.py` | Stat arb (Sharpe 4.0+) |
| **Meta-Labeling** | `core/gold_standard_models.py` | Triple barrier confidence |
| **Quant Formulas** | `core/quant_formulas.py` | Avellaneda-Stoikov, Kelly |
| **Execution** | `core/nautilus_executor.py` | HFT order management |
| **Master Predictor** | `core/gold_standard_predictor.py` | Unified ensemble |

### TIER 1: Must-Have (GitHub Verified)

| Framework | Stars | Use |
|-----------|-------|-----|
| Time-Series-Library | 11.3k | iTransformer, TimeMixer, TimeXer |
| Microsoft Qlib | 35.4k | Multi-paradigm ML |
| Stable-Baselines3 | 12.5k | PPO, SAC, TD3 |
| Optuna | 13.3k | Hyperparameter optimization |

### TIER 2: Specialized

| Framework | Stars | Use |
|-----------|-------|-----|
| mlfinlab | 4.5k | Triple Barrier, Meta-Labeling |
| hmmlearn | - | Renaissance regime detection |
| pykalman | - | Goldman Kalman filters |
| XGBoost/LightGBM/CatBoost | 25k+ | Gradient boosting ensemble |

### TIER 3: Chinese Quant (Gitee)

| Framework | Use |
|-----------|-----|
| QUANTAXIS | Rust-accelerated (570ns/op) |
| vnpy | Event-driven trading |
| Attention Factors | Stat arb (Sharpe 4.0+) |

### Gold Standard Libraries
```bash
pip install -r requirements_institutional.txt
```

### Signal Categories (110+ Total)
- **Alpha101**: 62 WorldQuant alphas
- **Renaissance**: 50 weak signals (trend, mean-rev, momentum, vol, microstructure)
- **Cross-Asset**: DXY, VIX, SPX, Gold, Oil signals

### Workflow (Always Follow)
1. **Backtest first**: `python backtest.py --strategy ml_ensemble --start 2026-01-01 --end 2026-01-08`
2. **Validate performance**: Win rate >55%, Sharpe >1.5, Max DD <5%
3. **Paper trade 2 weeks**: `python start_trading.py --mode paper --session auto`
4. **Go live**: `python start_trading.py --mode live --session morning`

### Usage
```python
from core.institutional_predictor import create_predictor
from core.quant_formulas import KellyCriterion, AvellanedaStoikov

# Load predictor (HMM + Kalman + Ensemble)
predictor = create_predictor(Path("models/institutional"))
result = predictor.predict("EURUSD", data)
# Returns: signal, confidence, regime, position_multiplier

# Position sizing with Kelly
kelly = KellyCriterion()
size = kelly.fractional_kelly(win_prob=0.55, win_loss_ratio=1.5, fraction=0.25)

# Market making quotes
mm = AvellanedaStoikov(gamma=0.1, sigma=0.02, k=1.5)
bid, ask = mm.optimal_quotes(mid_price=1.1000, inventory=10, time_remaining=0.5)
```

### Renaissance Weak Signals (50+)
**Categories:**
- Trend (10 signals): MA crossovers, slopes, alignment
- Mean Reversion (10 signals): Z-scores, Bollinger Bands, RSI
- Momentum (10 signals): ROC, MACD, acceleration
- Volatility (10 signals): ATR, vol percentiles, Keltner
- Microstructure (10 signals): Spread, order flow, reversals

**Usage:**
```python
from core.renaissance_signals import RenaissanceSignalGenerator

generator = RenaissanceSignalGenerator()
data_with_signals = generator.generate_all_signals(historical_data)
data_with_signals = generator.ensemble_signals(data_with_signals, method='average')

# Use ensemble_signal (-1, 0, 1) and ensemble_confidence (0-1)
```

### Decision Priority
1. **Backtest results** > intuition
2. **ML predictions** > manual signals
3. **Ensemble** > single model
4. **Statistical significance** > anecdotal evidence
5. **Risk management** > maximum returns

**Before live trading:** ALWAYS backtest, ALWAYS paper trade, ALWAYS validate performance.

---

## Session Startup

**Before working, ensure SSH tunnel is running:**
```powershell
powershell -File C:\Users\kevin\forex\scripts\start_oracle_tunnel.ps1
```

This forwards:
- `localhost:37777` → Oracle Cloud claude-mem (auto-capture)
- `localhost:3847` → Oracle Cloud memory-keeper (manual saves)

## Permission Controls

**Toggle bypass permissions during session:**
- Press `Ctrl+P` to toggle on/off instantly
- Bypass mode = No confirmations (fast vibe coding)
- Ask mode = Confirm each action (safe review)

**Toggle before session starts:**
```bash
python scripts/toggle_permissions.py bypass   # Fast mode
python scripts/toggle_permissions.py ask      # Safe mode
python scripts/toggle_permissions.py status   # Check current
```

**Default:** Bypass mode (no confirmations) - See `.claude/rules/permissions.md` for details

## Project Isolation (CRITICAL)

| Rule | Value |
|------|-------|
| **ONLY** access paths | containing `/forex/` |
| **NEVER** access | `/kalshi/`, `/bitcoin/`, `/polymarket/`, `/alpaca/`, `/paymentcrypto/` |
| **projectDir** | `C:\Users\kevin\forex` |
| **memory-keeper project** | `forex` |

**Violations contaminate shared infrastructure and break all 6 projects.**

---

## Project Overview

High-frequency forex trading system using machine learning for price prediction and strategy optimization.

## Core Architecture

- **Data Pipeline**: Real-time forex data from multiple providers (TrueFX, Dukascopy, Hostinger)
- **Data Storage**: Oracle Cloud (89.168.65.47) - `/home/ubuntu/projects/forex/`
  - Local: Minimal working data only
  - Remote: `data/` (raw), `data_cleaned/` (processed), `models/`, `logs/`
- **ML Models**: Time series forecasting models in `models/`
- **Training Framework**: Time-Series-Library for model experimentation
- **Compute**: Oracle Cloud for training and MCP servers (offloads local machine)

## Remote Infrastructure Architecture

**Local machine runs as THIN CLIENT only:**
```
Local Machine (Minimal Load)
├── Claude CLI only
└── SSH tunnel to Oracle Cloud

Oracle Cloud (89.168.65.47) - /home/ubuntu/projects/forex/
├── MCP Servers (remote execution)
├── Data storage (data/, data_cleaned/, models/, logs/)
└── Training compute (remote_train.py)

GitHub (forex-memory repo)
└── Memory/Context storage (replaces local SQLite/Chroma)
```

**Why remote:**
- Local machine cannot handle MCP servers + context storage
- Oracle Cloud: 42GB storage, 12GB RAM, ARM64 compute
- GitHub: Free private repo for memory persistence
- Zero local resource usage beyond Claude CLI

## Dual Memory System (REMOTE EXECUTION)

This project uses **two complementary memory systems**, both running **remotely on Oracle Cloud**:

### 1. Claude-Mem Plugin (Automatic)
- **Runs on**: Oracle Cloud (89.168.65.47)
- **Auto-captures** all coding sessions via hooks
- AI-compressed observations stored in GitHub repository
- Web viewer at Oracle Cloud (tunneled if needed)
- 3-layer search: search() → timeline() → get_observations()
- **Zero manual effort** - captures everything automatically
- **Storage**: GitHub repository (not local)

### 2. Memory-Keeper MCP (Manual Control)
- **Runs on**: Oracle Cloud (89.168.65.47)
- **Storage**: GitHub private repository (`forex-memory`)
- **Explicit** context management with manual save/search
- Structured by categories, priorities, channels
- Knowledge graph with relationships
- Checkpoints and session branching
- **High-value items only** - you choose what to save
- **Zero local storage** - all in GitHub

**When to use which:**
- **Claude-mem**: Passive background capture (automatic)
- **Memory-keeper**: Critical decisions, architecture, configs (manual)
- **Both**: Different retrieval strategies for different needs

**Architecture:**
```
Claude Code (local)
  → SSH tunnel
  → MCP Servers (Oracle Cloud)
  → GitHub API
  → Memory stored in GitHub repo
```

**Setup:** See `GITHUB_MEMORY_SETUP.md` for initial configuration

### Memory-Keeper Required Actions

**At session start:**
```
mcp__memory-keeper__context_session_start
- projectDir: C:\Users\kevin\forex
- name: Descriptive session name
- description: What you're working on
```

### Claude-Mem Search Workflow

**3-layer retrieval (always follow this order):**
```
1. search(query) → Get index with IDs (~50-100 tokens/result)
2. timeline(anchor=ID) → Get context around interesting results
3. get_observations([IDs]) → Fetch full details ONLY for filtered IDs
```

**NEVER fetch full details without filtering first - 10x token savings**

**During work - ALWAYS save:**
- Architecture decisions → `category: decision, priority: high`
- Completed tasks → `category: progress`
- Errors and solutions → `category: error, priority: high`
- Training configs/results → `category: note, priority: high`
- API configurations → `category: note, priority: high`

**Link related items:**
```
mcp__memory-keeper__context_link
- Link data sources to processing scripts
- Link training scripts to model outputs
- Link errors to solutions
```

**Before major changes:**
```
mcp__memory-keeper__context_checkpoint
- includeGitStatus: true
- includeFiles: true
```

**Search context when needed:**
```
mcp__memory-keeper__context_search
- Use natural language queries
- Filter by category, channel, priority
```

## Project-Specific Guidelines

### Data Sources
- TrueFX: Real-time tick data
- Dukascopy: Historical data
- Hostinger: Custom data storage
- Track all API endpoints and credentials in memory (category: decision)

### Environment Files
- `.env`: Main configuration with API keys
- `.env.backup`: Backup copy
- `.env_hostinger`: Hostinger-specific config
- NEVER commit these files
- Always save env structure to memory when adding new keys

### Data Processing
- Raw data → `data/`
- Cleaned data → `data_cleaned/`
- Always document cleaning steps in memory (category: decision)
- Link: raw_data → cleaning_script → cleaned_data → training_data

### Model Training
- **Run training on Oracle Cloud** to offload local compute
- Use `remote_train.py` for remote execution
- Save hyperparameters to memory before training (category: note, priority: high)
- Link training runs to their datasets
- Document performance metrics (category: progress)
- Save failed experiments and why (category: error)
- Models stored on Oracle Cloud, pull locally only when needed

### Code Organization
- Keep root directory clean (only folders and .env files)
- Scripts should be organized in subdirectories
- Delete temporary/test scripts after use

## Communication Preferences

Based on `.claude/settings.json`:
- Maximum automation - don't ask for permission
- Act first, explain after
- Succinct, direct responses
- No emojis
- Production-ready code only
- No placeholders or TODOs in code

## Context Preservation Strategy (CRITICAL)

**Vibe coding = long sessions = context compaction risk**

### Proactive Memory Saving

**Save immediately, not in batches:**
- After every important decision (don't wait)
- After completing any task
- After solving any error
- After discovering anything important

**NEVER rely solely on conversation history:**
- Conversation will be compacted
- Only memory-keeper persists
- Save to memory = survives compaction

### Before Compaction (High Token Usage)

**When context grows large, ALWAYS run:**
```
mcp__memory-keeper__context_prepare_compaction
```
This auto-saves critical context before compaction happens.

**Also create checkpoint:**
```
mcp__memory-keeper__context_checkpoint
- name: "pre_compaction_[timestamp]"
- includeGitStatus: true
- includeFiles: true
```

### Must Survive Compaction (Priority: High)

**Always save these with priority: high:**
1. Current work-in-progress and next steps
2. Recent architecture decisions
3. Active errors/blockers and their solutions
4. Configuration changes made this session
5. File locations of important scripts/data
6. What was tried and didn't work

### Recovery After Compaction

**If context was lost:**
```
1. mcp__memory-keeper__context_search_all
   - query: "recent decisions and progress"
   - createdAfter: [last session start]

2. mcp__memory-keeper__context_timeline
   - groupBy: "hour"
   - relativeTime: "today"

3. mcp__memory-keeper__context_diff
   - since: [last checkpoint]
```

### Regular Checkpointing Schedule

**Create checkpoints:**
- Every major feature/task completion
- Before risky changes or experiments
- Every 50-100 messages in long sessions
- Before and after model training runs
- When switching between different tasks

## Workflow Expectations

1. **Session Start**:
   - Initialize memory-keeper session (runs on Oracle Cloud automatically)
   - Claude-mem auto-captures (runs on Oracle Cloud)
   - Both connect via SSH tunnel transparently
2. **Before Coding**: Search BOTH systems for relevant context
   - Claude-mem: `search()` for past work, patterns, solutions
   - Memory-keeper: `context_search()` for decisions, configs
   - **Searches execute on Oracle Cloud**, results returned to you
3. **During Work**:
   - Claude-mem: Auto-captures (no action needed, runs remotely)
   - Memory-keeper: Save HIGH-VALUE items IMMEDIATELY (decisions, configs, critical errors)
   - **All saves go to GitHub repository** (not local storage)
4. **Link Knowledge**: Connect related items in memory-keeper knowledge graph
5. **Before Major Changes**: Create memory-keeper checkpoint (stored in GitHub)
6. **On Errors**: Auto-captured by claude-mem, save critical ones to memory-keeper
7. **High Token Usage**: Run prepare_compaction + checkpoint
8. **Regular Intervals**: Checkpoint every 50-100 messages

**Important:** From your perspective, workflow is unchanged. MCP servers on Oracle Cloud + GitHub storage are transparent. You interact the same way, but zero local resources used.

## Knowledge Graph Structure

Use these relationships when linking:
- `depends_on`: Script/model depends on data/config
- `implements`: Code implements a decision/design
- `blocks`: Error blocks progress
- `references`: Documentation references code
- `related_to`: General relationships

## Quality Standards

- Modern Python patterns
- Type hints where appropriate
- Error handling at system boundaries only
- No over-engineering
- No unnecessary abstractions
- Production-ready on first implementation

## Quick Reference

**Oracle Cloud Operations:**
```bash
# Sync data to cloud
python3 scripts/oracle_sync.py push all

# Pull models from cloud
python3 scripts/oracle_sync.py pull models

# Check cloud storage
python3 scripts/oracle_sync.py status

# SSH to cloud
bash scripts/oracle.sh ssh

# Run training remotely
python3 scripts/remote_train.py run path/to/train.py
```

**MCP Server Management:**
```bash
# Check MCP server status (on Oracle Cloud)
bash scripts/oracle.sh ssh "/home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh status"

# View MCP logs
bash scripts/oracle.sh ssh "/home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh logs-memory"

# Restart MCP server
bash scripts/oracle.sh ssh "/home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh restart-memory"
```

**Documentation:**
- `ORACLE_CLOUD.md` - Data storage and sync guide
- `GITHUB_MEMORY_SETUP.md` - Memory migration setup
- `MCP_MIGRATION_SUMMARY.md` - Architecture overview
- `.claude/rules/remote-architecture.md` - Detailed architecture
- `PROJECT_REGISTRY.md` - Multi-project isolation
