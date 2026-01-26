# ML Training Guidelines

## Current Performance (2026-01-14)

| Symbol | Accuracy | AUC | Edge |
|--------|----------|-----|------|
| EURUSD | **74.42%** | 0.8174 | +24.42% |
| GBPUSD | **72.69%** | 0.7985 | +22.69% |
| USDJPY | **70.57%** | 0.7730 | +20.57% |

**Target achieved: 70%+ accuracy with full GPU ensemble (XGBoost + LightGBM + CatBoost)**

### Data Available for Expansion (78 pairs in Dukascopy)
- **Priority 1 (Majors):** EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Priority 2 (Crosses):** EURJPY, GBPJPY, EURGBP, EURCHF, AUDJPY, EURAUD, GBPAUD
- **Priority 3 (More):** EURNZD, GBPNZD, AUDNZD, NZDJPY, AUDCAD, CADCHF, CADJPY

---

## PRIMARY: Local RTX 5080 Training

**$3000 RTX 5080 dedicated to forex ML. Use it to the MAX.**

### Default Mode: Hybrid Retraining

Bot automatically retrains every 120 seconds using historical + live data:

```bash
# Start bot with hybrid retraining (auto-enabled)
python main.py trade --mode paper --symbols EURUSD,GBPUSD,USDJPY

# Or direct script
python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD,USDJPY
```

**What happens:**
1. Load 50k historical samples (from 200k+ training data)
2. Combine with live tick data (weighted 3x)
3. Train XGBoost + LightGBM on GPU (80-95% util)
4. Hot-swap if accuracy improves
5. Trading continues uninterrupted

### Batch Training (Full Retrain)

```bash
cd C:\Users\kevin\forex\forex
venv\Scripts\activate
python main.py train
# Or: python training_package/train_models.py
```

### GPU Config (CRITICAL)

```python
# core/gpu_config.py - DO NOT use CPU settings
'tree_method': 'hist',      # XGBoost 2.0+ (NOT 'gpu_hist')
'device': 'cuda',           # This triggers GPU
'task_type': 'GPU',         # CatBoost GPU mode
'max_depth': 12,            # NOT 6 or 8
```

**Common mistakes:**
- `tree_method='gpu_hist'` - WRONG in XGBoost 2.0+
- `task_type='CPU'` - WRONG, use 'GPU'
- `max_depth=6` - TOO SHALLOW for 16GB VRAM

### Models Trained

- XGBoost (GPU-accelerated, depth=12)
- LightGBM (GPU-accelerated, num_leaves=511)
- CatBoost (GPU-accelerated, depth=12)
- Ensemble voting (average of all)

### Targets Available

- `target_direction_N`: Binary direction (N = 1,5,10,20,50,100 ticks)
- `target_return_N`: Log return in bps
- `target_vol_N`: Forward volatility

### Output Location

```
models/production/              # Canonical location (2026-01-16)
├── EURUSD_models.pkl
├── EURUSD_results.json
├── GBPUSD_models.pkl
├── GBPUSD_results.json
├── USDJPY_models.pkl
└── USDJPY_results.json

models/hft_ensemble/            # Alias (backward-compatible)
```

---

## Edge Calculation

**Edge = Accuracy - 50% (random baseline)**

| Accuracy | Edge | Expected Win Rate |
|----------|------|-------------------|
| 50% | 0% | Break-even |
| 55% | 5% | Slight profit |
| 60% | 10% | Good profit |
| 65% | 15% | Strong profit |
| **70%** | **20%** | **Excellent** |
| **73%** | **23%** | **Current EURUSD** |

**With 70%+ accuracy:**
- Expected win rate: 70%
- Kelly optimal bet: ~40% of bankroll
- Using 25% Kelly = conservative but steady growth

---

## Historical Data (Foundation)

| Symbol | Samples | Location |
|--------|---------|----------|
| EURUSD | 157,619 | `training_package/EURUSD/` |
| GBPUSD | 184,688 | `training_package/GBPUSD/` |
| USDJPY | 220,528 | `training_package/USDJPY/` |

**Total: 562,835 samples** with 100+ features each

---

## Validation Requirements

**Before training:**
1. Walk-forward CV with `gap >= 10` periods
2. Check overfit ratio: `train_score / test_score < 1.15`
3. Validate on held-out period (last 20%)

**Before paper trading:**
1. Accuracy > 60%
2. AUC > 0.65
3. Sharpe ratio > 1.5

**Before live trading:**
1. Paper trade minimum 24 hours
2. Compare paper vs backtest metrics
3. Accuracy sustained above 65%

---

## Key Metrics

| Metric | Target | Excellent |
|--------|--------|-----------|
| Accuracy | >65% | >70% |
| AUC | >0.70 | >0.80 |
| Win Rate | >58% | >65% |
| Sharpe | >1.5 | >2.5 |
| Max Drawdown | <10% | <5% |

---

## Troubleshooting

### "GPU not being used"

Check nvidia-smi during training:
```bash
nvidia-smi -l 1
```

If GPU util is low:
1. Verify `device='cuda'` in gpu_config.py
2. Verify XGBoost using `tree_method='hist'` (not 'gpu_hist')
3. Increase `max_depth` and `num_leaves`

### "Accuracy dropping"

1. Check for data drift (market regime change)
2. Increase historical sample weight
3. Reduce retrain interval (more frequent updates)

### "Models not hot-swapping"

1. Check that new accuracy > previous best
2. Verify ensemble has valid predictions
3. Check logs for errors during training
