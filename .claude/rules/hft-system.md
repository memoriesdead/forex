# HFT System Rules

## Architecture Overview

**Renaissance-Level HFT System** built 2026-01-10, **GPU Maximized** 2026-01-14, **51 Pairs Trained** 2026-01-17

### Current State (2026-01-17)
| Metric | Value |
|--------|-------|
| **Pairs Trained** | 51 |
| **Average Accuracy** | 63.48% |
| **Features per Pair** | 575 (mega features) |
| **Models per Pair** | 3 (XGBoost + LightGBM + CatBoost) |
| **Edge vs Random** | +13.48% |

```
┌─────────────────────────────────────────────────────────────────┐
│                    HFT TRADING SYSTEM                           │
│              RTX 5080 (16GB) - 80-95% GPU Utilization           │
├─────────────────────────────────────────────────────────────────┤
│  Entry Point                                                    │
│  └── main.py (trade|train|backtest|status|data)                 │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                     │
│  ├── TrueFX (live streaming)                                    │
│  ├── LiveTickBuffer (50k ticks, ring buffer)                    │
│  └── hft_data_loader.py (unified loader)                        │
├─────────────────────────────────────────────────────────────────┤
│  Feature Layer (100+ features)                                  │
│  ├── Alpha101 (62 WorldQuant alphas)                            │
│  ├── Renaissance (50 weak signals)                              │
│  ├── Order Flow (OFI, OEI, VPIN, Hawkes)                        │
│  ├── Microstructure (TSRV, noise, SNR)                          │
│  └── hft_feature_engine.py (unified engine)                     │
├─────────────────────────────────────────────────────────────────┤
│  Prediction Layer (GPU-ACCELERATED)                             │
│  ├── XGBoost (device=cuda, depth=12)                            │
│  ├── LightGBM (device=gpu, num_leaves=511)                      │
│  ├── CatBoost (task_type=GPU, border_count=32)                  │
│  └── HotSwappableEnsemble (atomic model swap)                   │
├─────────────────────────────────────────────────────────────────┤
│  Hybrid Retraining Layer (2026-01-14)                           │
│  ├── HybridRetrainer → 50k hist + live (3x weighted)            │
│  ├── Retrain every 120s on GPU (80-95% util)                    │
│  ├── Hot-swap if accuracy improves                              │
│  └── Target: 70%+ accuracy (ACHIEVED)                           │
├─────────────────────────────────────────────────────────────────┤
│  Execution Layer                                                │
│  ├── IB Gateway (localhost:4004, SSH tunnel)                    │
│  ├── Latency simulation                                         │
│  ├── Queue position tracking                                    │
│  └── Fill probability estimation                                │
├─────────────────────────────────────────────────────────────────┤
│  Risk Layer                                                     │
│  ├── Kelly Criterion (25% fraction)                             │
│  ├── Max drawdown (5%)                                          │
│  ├── Daily trade limit (100)                                    │
│  └── Position limit (2% per trade)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Run trading (preferred method)
python main.py trade --mode paper --symbols EURUSD,GBPUSD,USDJPY

# Check status
python main.py status

# Train models
python main.py train

# Backtest
python main.py backtest --strategy ml_ensemble --days 30
```

## Core Files

### Entry Point

| File | Purpose |
|------|---------|
| `main.py` | **Single entry point** - CLI for all operations |

### HFT Core Modules (Reorganized 2026-01-16)

| Module | File | Key Classes |
|--------|------|-------------|
| **core/ml/** | `gpu_config.py` | `get_xgb_gpu_params`, `get_lgb_gpu_params`, `get_catboost_gpu_params` |
| | `retrainer.py` | `HybridRetrainer`, `get_hybrid_retrainer` |
| | `ensemble.py` | `HotSwappableEnsemble`, `ModelVersion` |
| | `live_retrainer.py` | `LiveRetrainer`, `get_retrainer` |
| **core/data/** | `loader.py` | `UnifiedDataLoader`, `TrueFXLiveLoader` |
| | `buffer.py` | `LiveTickBuffer`, `TickRecord` |
| **core/features/** | `engine.py` | `HFTFeatureEngine`, `FastTechnicalFeatures` |
| | `alpha101.py` | `Alpha101Complete` |
| | `renaissance.py` | `RenaissanceSignalGenerator` |
| **core/execution/** | `order_book.py` | `OrderBookL3`, `PriceLevel`, `Order` |
| | `queue_position.py` | `RiskAverseQueueModel`, `ProbabilisticQueueModel` |
| | `fill_probability.py` | `PoissonFillModel`, `MarketImpactModel` |

### HFT Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `hft_trading_bot.py` | Main trading bot |
| `train_parallel_max.py` | **Pipeline parallel training (3x faster)** |
| `train_all_mega.py` | Sequential mega feature training |
| `prepare_hft_training.py` | Package training data |
| `start_hft.py` | Legacy controller (use main.py instead) |

## Current Performance (2026-01-14)

| Symbol | Accuracy | AUC | Status |
|--------|----------|-----|--------|
| EURUSD | **73.24%** | 0.8071 | Live |
| GBPUSD | **71.46%** | 0.7853 | Live |
| USDJPY | **70.38%** | 0.7671 | Live |

**Edge:** 70%+ accuracy = 20%+ edge over random (50%)

## Hybrid Retraining (NEW)

**How it works:**
1. Load 50k historical samples (from 200k+ total)
2. Combine with live tick data (weighted 3x)
3. Train XGBoost + LightGBM on GPU every 120s
4. Hot-swap models if accuracy improves
5. Trading continues uninterrupted

**Why it works:**
- Historical data provides diverse market conditions
- Live data captures current market behavior
- Weighted combination adapts to regime changes

## GPU Configuration

```python
# XGBoost (core/gpu_config.py)
'tree_method': 'hist',    # XGBoost 2.0+
'device': 'cuda',         # GPU acceleration
'max_depth': 12,          # Deeper for 16GB VRAM

# LightGBM
'device': 'gpu',
'num_leaves': 511,        # 2^12 - 1
'histogram_pool_size': 1024,

# CatBoost
'task_type': 'GPU',
'border_count': 32,       # GPU optimization
'depth': 12,
```

## Risk Management

| Limit | Default |
|-------|---------|
| Max position per trade | 2% of account |
| Max drawdown | 5% |
| Daily trade limit | 100 |
| Kelly fraction | 25% |

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Accuracy | >70% | **71-73%** |
| AUC | >0.75 | **0.77-0.81** |
| Win Rate | >60% | Measuring |
| Sharpe | >2.0 | Measuring |
| Max Drawdown | <5% | Measuring |

## Files to NEVER Modify

- `core/order_book_l3.py` - Core order book logic
- `core/queue_position.py` - Queue models from HftBacktest
- `core/fill_probability.py` - Academic fill models

**If changes needed, create new file and extend.**
