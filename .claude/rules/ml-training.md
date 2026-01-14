# ML Training Guidelines (Gold Standard)

## CRITICAL: Target 70%+ Accuracy

**Renaissance was 50.75% in 1980. Modern ML on H100 targets 70%+.**

---

## HFT Training Workflow (NEW - 2026-01-10)

**Preferred method for HFT models:**

### 1. Prepare Training Data
```bash
python scripts/start_hft.py prepare --symbols EURUSD,GBPUSD,USDJPY --days 30
```

Creates `training_package/` with:
- Parquet files (train/val/test splits)
- 100+ features per tick
- Training script for Vast.ai
- Requirements and run script

### 2. Train on Vast.ai H100
```bash
# Rent H100 (~$2.50/hr)
scp -r training_package/* root@<VAST_IP>:/workspace/
ssh root@<VAST_IP>
./run_training.sh

# Download (after ~1-2 hours)
scp root@<VAST_IP>:/workspace/hft_models.tar.gz .
tar -xzvf hft_models.tar.gz -C models/hft_ensemble/
```

### 3. Models Trained
- XGBoost (GPU-accelerated)
- LightGBM (GPU-accelerated)
- CatBoost (GPU-accelerated)
- Ensemble voting

### 4. Targets Available
- `target_direction_N`: Binary direction (N = 1,5,10,20,50,100 ticks)
- `target_return_N`: Log return in bps
- `target_vol_N`: Forward volatility

### 5. Output Location
```
models/hft_ensemble/
├── EURUSD_target_direction_10_models.pkl
├── GBPUSD_target_direction_10_models.pkl
├── USDJPY_target_direction_10_models.pkl
└── *_results.json
```

---

## Gold Standard Frameworks (Verified Audit)

### TIER 1: Must-Have (High Stars, Active)

| Framework | Stars | Use For |
|-----------|-------|---------|
| **Time-Series-Library** | 11.3k | 47 SOTA forecasting (iTransformer, TimeMixer, TimeXer) |
| **Microsoft Qlib** | 35.4k | Multi-paradigm ML pipeline |
| **Stable-Baselines3** | 12.5k | RL agents (PPO, SAC, TD3) |
| **FinRL** | 13.7k | Financial RL, cloud training |
| **Optuna** | 13.3k | Hyperparameter optimization |

### TIER 2: Specialized

| Framework | Stars | Use For |
|-----------|-------|---------|
| **MlFinLab** | 4.5k | Triple Barrier, Fractional Diff |
| **NautilusTrader** | 17.2k | HFT execution (sub-microsecond) |
| **HftBacktest** | 3.5k | Tick-level validation |
| **hmmlearn** | - | HMM regime detection (Renaissance) |
| **pykalman** | - | Kalman Filter (Goldman) |

### TIER 3: Chinese Quant

| Framework | Use For |
|-----------|---------|
| **QUANTAXIS** | Rust-accelerated backtesting (570ns/operation) |
| **vnpy** | Event-driven trading engine |
| **Quantformer** | Transformer for finance |
| **Attention Factors** | Stat arb (Sharpe 4.0+) |

## Training Location

**NEVER train locally. ALWAYS use Vast.ai H100:**
```bash
python scripts/train_all_frameworks_h100.py
```

## Models to Train Per Pair

1. **HMM** (Renaissance) - 3-state regime detection
2. **Kalman Filter** (Goldman) - Dynamic mean estimation
3. **XGBoost + LightGBM + CatBoost** - Gradient boosting ensemble
4. **PPO + SAC** - RL agents for position sizing
5. **iTransformer + DLinear** - SOTA time series forecasting
6. **Optuna** - Hyperparameter optimization
7. **Weighted Ensemble** - All models combined

## Before Training

**Save to memory:**
1. Hyperparameters (category: note, priority: high)
2. Dataset version/path (category: note, priority: high)
3. Model architecture choices (category: decision)
4. Training objective (category: task)

**Create checkpoint:**
```
mcp__memory-keeper__context_checkpoint
```

## Vast.ai Workflow

1. **Prepare data** - Package cleaned data
2. **Rent H100** - ~$2-3/hour
3. **Upload data** - SCP to instance
4. **Run training** - 1-2 hours
5. **Download models** - SCP back
6. **Destroy instance** - Stop billing
7. **Sync to Oracle Cloud** - For live trading

## After Training

**Save to memory:**
1. Final metrics (category: progress, priority: high)
2. Model path: `models/gold_standard/` (category: note)
3. What worked (category: decision)
4. Validation accuracy per model (category: progress)

**Model files:**
```
models/gold_standard/
├── EURUSD_gold_standard.pkl
├── GBPUSD_gold_standard.pkl
├── USDJPY_gold_standard.pkl
├── ... (per pair)
└── all_pairs_gold_standard.pkl
```

## Key Metrics

**Target performance:**
- Accuracy: >70% (modern ML target)
- Win rate: >60%
- Sharpe ratio: >2.0
- Max drawdown: <5%

**Per model validation:**
- XGBoost: Target 65%+
- LightGBM: Target 65%+
- CatBoost: Target 65%+
- iTransformer: Target 60%+
- Ensemble: Target 70%+

## Credit Check

**Before training, verify Vast.ai credit:**
- Current: ~$113 available
- H100: ~$2.50/hour
- Estimated training: 1-2 hours = $3-5

## Time-Series-Library Models

**Priority order:**
1. **iTransformer** - ICLR 2024, best for multivariate
2. **TimeMixer** - ICLR 2024, multi-scale mixing
3. **TimeXer** - NeurIPS 2024, exogenous factors
4. **TimesNet** - ICLR 2023, temporal 2D variation
5. **PatchTST** - ICLR 2023, patching
6. **DLinear** - AAAI 2023, simple but effective

## RL Algorithms

**From Stable-Baselines3:**
- **PPO** - Proximal Policy Optimization (most stable)
- **SAC** - Soft Actor-Critic (continuous action)
- **TD3** - Twin Delayed DDPG
- **A2C** - Advantage Actor-Critic

---

## Core Modules (Implemented 2026-01-10)

### Signal Generation

| Module | File | Features |
|--------|------|----------|
| **Alpha101** | `core/alpha101_complete.py` | 62 WorldQuant alphas |
| **Renaissance Signals** | `core/renaissance_signals.py` | 50+ weak signals |
| **Cross-Asset** | `core/cross_asset_signals.py` | DXY, VIX, SPX, Gold, Oil |

### Walk-Forward Validation

| Method | Class | Use Case |
|--------|-------|----------|
| **Expanding Window** | `WalkForwardCV(expanding=True)` | Default for training |
| **Rolling Window** | `WalkForwardCV(expanding=False)` | Regime changes |
| **Purged K-Fold** | `PurgedKFold` | Overlapping labels |
| **CPCV** | `CombinatorialPurgedCV` | Multiple backtests |

**File:** `core/walk_forward.py`

**Usage:**
```python
from core.walk_forward import expanding_window_cv, purged_cv

cv = expanding_window_cv(n_splits=5, gap=10)
for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

### Prediction Pipeline

| Module | File | Purpose |
|--------|------|---------|
| **Gold Standard Predictor** | `core/gold_standard_predictor.py` | Master ensemble |
| **Gold Standard Models** | `core/gold_standard_models.py` | iTransformer, TimeXer, Attention |
| **Institutional Predictor** | `core/institutional_predictor.py` | HMM, Kalman, Ensemble |
| **Nautilus Executor** | `core/nautilus_executor.py` | HFT execution |

### Alpha101 Usage

```python
from core.alpha101_complete import Alpha101Complete

alpha = Alpha101Complete()
df_with_alphas = alpha.generate_all_alphas(df)
# Returns 62 alpha columns: alpha_001 through alpha_062
```

### Cross-Asset Signals Usage

```python
from core.cross_asset_signals import CrossAssetSignalGenerator

generator = CrossAssetSignalGenerator()
signals = generator.generate_all_signals(
    forex_data=eurusd_df,
    cross_asset_data={'DXY': dxy_df, 'VIX': vix_df, 'SPX': spx_df}
)
# Returns: dxy_*, vix_*, spx_*, gold_*, oil_*, risk_sentiment, cross_asset_signal
```

---

## Feature Engineering Pipeline

**Order of signal generation:**

1. **Base Features** (OHLCV + returns + volatility)
2. **Alpha101** (62 alphas from `alpha101_complete.py`)
3. **Renaissance Signals** (50+ from `renaissance_signals.py`)
4. **Cross-Asset Signals** (from `cross_asset_signals.py`)
5. **Ensemble Signal** (combined weighted average)

**Total features:** 150+ signals before model training

---

## Validation Requirements (MANDATORY)

**Before training ANY model:**
1. Run walk-forward CV with `gap >= 10` periods
2. Check overfit ratio: `train_score / test_score < 1.15`
3. Validate on held-out period (last 20% of data)

**Before paper trading:**
1. Backtest on minimum 1 year of data
2. Sharpe ratio > 1.5
3. Max drawdown < 5%
4. Win rate > 55%

**Before live trading:**
1. Paper trade minimum 2 weeks
2. Compare paper vs backtest metrics
3. Overfit ratio < 1.10
