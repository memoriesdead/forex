# RTX 5080 GPU Maximization Rules

## Hardware Context

**$3000+ RTX 5080 dedicated to forex ML trading.**

| Spec | Value |
|------|-------|
| GPU | RTX 5080 (Blackwell) |
| VRAM | 16GB GDDR7 |
| CUDA Cores | 10,752 |
| Tensor Cores | 336 |
| Target Utilization | 80-95% during training |

---

## CRITICAL: GPU Settings

### XGBoost - ALWAYS use gpu_hist

```python
# CORRECT
'tree_method': 'gpu_hist',
'device': 'cuda',
'predictor': 'gpu_predictor',

# WRONG - runs on CPU!
'tree_method': 'hist',
```

### CatBoost - ALWAYS use GPU

```python
# CORRECT
'task_type': 'GPU',
'devices': '0',

# WRONG - runs on CPU!
'task_type': 'CPU',
```

### LightGBM - ALWAYS use GPU

```python
# CORRECT
'device': 'gpu',
'gpu_platform_id': 0,
'gpu_device_id': 0,

# WRONG - runs on CPU!
'device': 'cpu',
```

---

## Tree Depths for 16GB VRAM

| Model | Depth | Why |
|-------|-------|-----|
| XGBoost | 12 | Deep trees, 16GB can handle |
| LightGBM | 12 + num_leaves=511 | 2^12-1 leaves |
| CatBoost | 10 | GPU limit, border_count=32 |

**Never use depth=6 or depth=8** - wastes VRAM.

---

## Live Retraining Architecture

### Components

1. **LiveTickBuffer** (`core/live_tick_buffer.py`)
   - Ring buffer: 50,000 ticks
   - Thread-safe with RLock
   - Stores features + outcomes

2. **HotSwappableEnsemble** (`core/hot_swappable_ensemble.py`)
   - Atomic model replacement
   - Fallback to previous version
   - Version tracking

3. **LiveRetrainer** (`core/live_retrainer.py`)
   - Runs every 60 seconds
   - Trains XGB + LGB + CB on GPU
   - Hot-swaps if accuracy +0.5%

### Data Flow

```
Tick → Features → Buffer (50k)
                     ↓
            Every 60s: Extract 5000 ticks
                     ↓
            Train on GPU (80-95% util)
                     ↓
            Validate on holdout
                     ↓
            If better → Hot-swap
```

---

## Verification Commands

### Check GPU Utilization
```powershell
nvidia-smi -l 1
```

### Expected Pattern
- **3-7%**: Normal trading (predictions fast)
- **80-95%**: Every 60s burst (retraining)
- **0%**: No training happening (bad)

### Check CUDA Available
```python
import torch
print(torch.cuda.is_available())  # Must be True
print(torch.cuda.get_device_name(0))  # RTX 5080
```

---

## When Writing ML Code

### Always Import GPU Config
```python
from core.ml.gpu_config import (
    get_xgb_gpu_params,
    get_lgb_gpu_params,
    get_catboost_gpu_params,
    configure_gpu
)
```

### Never Hardcode CPU Settings
```python
# BAD - hardcoded
model = xgb.XGBClassifier(tree_method='hist')

# GOOD - use gpu_config
params = get_xgb_gpu_params()
model = xgb.XGBClassifier(**params)
```

---

## Troubleshooting

### GPU at 7% during training
**Cause:** Using CPU settings instead of GPU
**Fix:** Check `tree_method`, `task_type`, `device` params

### "CUDA out of memory"
**Cause:** Trees too deep or batch too large
**Fix:** Reduce max_depth or batch size

### Retraining never starts
**Cause:** Not enough labeled data
**Fix:** Wait 10+ min for buffer to fill (1000+ ticks)

### Models not improving
**Cause:** Market regime changed
**Fix:** Increase retrain frequency or add more features

---

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| GPU during training | 80-95% | 60%+ |
| Training time/symbol | 30-60s | <120s |
| Model accuracy | 65%+ | 60%+ |
| Win rate | 58%+ | 55%+ |

---

## GPU Core Modules (Reorganized 2026-01-16)

| File | Purpose |
|------|---------|
| `core/ml/gpu_config.py` | GPU-optimized params (depth=12, border_count=32) |
| `core/ml/retrainer.py` | Hybrid retrainer (historical + live) |
| `core/ml/ensemble.py` | HotSwappableEnsemble |
| `core/ml/live_retrainer.py` | Background GPU training |
| `core/data/buffer.py` | LiveTickBuffer ring buffer |
| `training_package/train_models.py` | Imports from core/ml/ |
| `scripts/hft_trading_bot.py` | Live retraining integration |
