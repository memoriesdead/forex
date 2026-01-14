# HFT System Rules

## Architecture Overview

**Renaissance-Level HFT System** built 2026-01-10

```
┌─────────────────────────────────────────────────────────────────┐
│                    HFT TRADING SYSTEM                           │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                     │
│  ├── TrueFX (historical tick data)                              │
│  ├── TrueFX (live streaming)                                    │
│  ├── Oracle Cloud (/home/ubuntu/market_data/)                   │
│  └── hft_data_loader.py (unified loader)                        │
├─────────────────────────────────────────────────────────────────┤
│  Feature Layer (100+ features)                                  │
│  ├── Alpha101 (62 WorldQuant alphas)                            │
│  ├── Renaissance (50 weak signals)                              │
│  ├── Order Flow (OFI, OEI, VPIN, Hawkes)                        │
│  ├── Microstructure (TSRV, noise, SNR)                          │
│  └── hft_feature_engine.py (unified engine)                     │
├─────────────────────────────────────────────────────────────────┤
│  Prediction Layer                                               │
│  ├── XGBoost (GPU-accelerated)                                  │
│  ├── LightGBM (GPU-accelerated)                                 │
│  ├── CatBoost (GPU-accelerated)                                 │
│  └── Ensemble voting                                            │
├─────────────────────────────────────────────────────────────────┤
│  Execution Layer                                                │
│  ├── IB Gateway (Oracle Cloud)                                  │
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

## Core Files

### HFT Core Modules (`core/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `order_book_l3.py` | Order book reconstruction | `OrderBookL3`, `PriceLevel`, `Order` |
| `queue_position.py` | Queue position tracking | `RiskAverseQueueModel`, `ProbabilisticQueueModel`, `L3FIFOQueueModel` |
| `fill_probability.py` | Fill estimation | `PoissonFillModel`, `MarketImpactModel`, `FillProbabilityEngine` |
| `tick_backtest.py` | Tick-level simulation | `TickBacktestEngine`, `Strategy` |
| `order_flow_features.py` | Order flow signals | `OrderFlowFeatures`, `TradeDirectionInference` |
| `latency_simulator.py` | Latency modeling | `LatencySimulator`, `LatencyConfig` |
| `microstructure_vol.py` | Noise filtering | `MicrostructureVolatility`, `JumpRobustVolatility` |
| `hft_data_loader.py` | Unified data loading | `UnifiedDataLoader`, `TrueFXLiveLoader` |
| `hft_feature_engine.py` | Feature generation | `HFTFeatureEngine`, `FastTechnicalFeatures` |

### HFT Scripts (`scripts/`)

| File | Purpose | Usage |
|------|---------|-------|
| `start_hft.py` | Master controller | `python scripts/start_hft.py [status|prepare|paper|live]` |
| `prepare_hft_training.py` | Vast.ai packager | `python scripts/prepare_hft_training.py --symbols EURUSD --days 30` |
| `hft_trading_bot.py` | Trading bot | `python scripts/hft_trading_bot.py --mode paper` |

## Workflow Rules

### 1. Data Preparation

**ALWAYS prepare data before training:**
```bash
python scripts/start_hft.py prepare --symbols EURUSD,GBPUSD,USDJPY --days 30
```

**Output:** `training_package/` directory with:
- `{SYMBOL}/train.parquet` - Training data
- `{SYMBOL}/val.parquet` - Validation data
- `{SYMBOL}/test.parquet` - Test data
- `{SYMBOL}/metadata.json` - Feature names, counts
- `train_hft_models.py` - Training script
- `requirements.txt` - Dependencies
- `run_training.sh` - Vast.ai runner

### 2. Training (Vast.ai ONLY)

**NEVER train locally. ALWAYS use Vast.ai H100:**

```bash
# 1. Rent H100 on vast.ai (~$2.50/hr)
# 2. Upload training package
scp -r training_package/* root@<VAST_IP>:/workspace/

# 3. SSH and run
ssh root@<VAST_IP>
./run_training.sh

# 4. Download models
scp root@<VAST_IP>:/workspace/hft_models.tar.gz .
tar -xzvf hft_models.tar.gz -C models/hft_ensemble/

# 5. DESTROY INSTANCE (stop billing!)
```

### 3. Paper Trading

**ALWAYS paper trade before live (24hr minimum):**
```bash
python scripts/start_hft.py paper --symbols EURUSD,GBPUSD
```

**Requirements:**
- SSH tunnel to Oracle Cloud (IB Gateway)
- Models in `models/hft_ensemble/`
- TrueFX credentials for live data

### 4. Live Trading

**Only after successful paper trading:**
```bash
python scripts/start_hft.py live --symbols EURUSD
```

**Requires typing "LIVE" to confirm.**

## Feature Engineering

### Using HFTFeatureEngine

```python
from core.hft_feature_engine import HFTFeatureEngine

# Initialize
engine = HFTFeatureEngine()

# Process historical data
engine.initialize(historical_df, symbol="EURUSD")

# Process live ticks
for tick in live_ticks:
    features = engine.process_tick(
        symbol="EURUSD",
        bid=tick.bid,
        ask=tick.ask,
        volume=tick.volume,
        timestamp=tick.timestamp
    )
    # features: Dict[str, float] with 100+ features
```

### Feature Categories

1. **Fast Technical** (20 features)
   - Returns at lags [1, 5, 10, 20]
   - Z-scores at windows [10, 20, 50]
   - Volatility estimates
   - Range positions
   - Volume ratios

2. **Alpha101** (62 features)
   - WorldQuant formulaic alphas
   - Cross-sectional signals

3. **Renaissance** (50 features)
   - Trend signals (MA, slopes)
   - Mean reversion (z-scores, bands)
   - Momentum (ROC, acceleration)
   - Volatility regime

4. **Order Flow** (10 features)
   - OFI (Order Flow Imbalance)
   - OEI (Order Execution Imbalance)
   - VPIN (Volume-synchronized PIN)
   - Hawkes intensity

5. **Microstructure** (8 features)
   - TSRV (Two Scales Realized Variance)
   - Noise variance
   - Signal-to-noise ratio

## Risk Management

### Built-in Limits

| Limit | Default | Configurable |
|-------|---------|--------------|
| Max position per trade | 2% of account | Yes |
| Max drawdown | 5% | Yes |
| Daily trade limit | 100 | Yes |
| Kelly fraction | 25% | Yes |

### Kelly Position Sizing

```python
from scripts.hft_trading_bot import RiskManager

risk = RiskManager(
    max_position_pct=0.02,
    max_drawdown_pct=0.05,
    kelly_fraction=0.25,
    max_daily_trades=100
)

# Check if can trade
can_trade, reason = risk.can_trade()

# Calculate position size
size = risk.calculate_position_size(signal, current_price)
```

## Latency Configuration

### Pre-configured Venues

| Venue | Feed Latency | Order Latency |
|-------|-------------|---------------|
| `retail_forex` | 50ms | 100ms |
| `institutional_forex` | 5ms | 10ms |
| `exchange_colocated` | 0.1ms | 0.5ms |
| `ib_gateway` | 30ms | 80ms |

### Usage

```python
from core.latency_simulator import LatencySimulator

sim = LatencySimulator(venue='ib_gateway')
delayed_time = sim.apply_feed_latency(tick_time)
order_received = sim.apply_order_latency(submit_time)
```

## Model Targets

### Classification Targets

- `target_direction_N`: 1 if price up after N ticks, 0 if down
- `target_up_N`: 1 if return > 0.5 bps after N ticks
- `target_down_N`: 1 if return < -0.5 bps after N ticks

### Regression Targets

- `target_return_N`: Log return in bps after N ticks
- `target_vol_N`: Realized volatility over next N ticks

### Horizons

Default: [1, 5, 10, 20, 50, 100] ticks

## Debugging

### Check System Status
```bash
python scripts/start_hft.py status
```

### Check IB Gateway
```bash
ssh ubuntu@89.168.65.47 "docker logs ibgateway | tail -20"
```

### Test Feature Engine
```python
from core.hft_feature_engine import HFTFeatureEngine
engine = HFTFeatureEngine()
# Engine self-tests on import
```

### Test Data Loader
```python
from core.hft_data_loader import UnifiedDataLoader
loader = UnifiedDataLoader()
data = loader.load_historical("EURUSD", start_date, end_date, source='truefx')
```

## Common Issues

### "No models found"
- Train models on Vast.ai first
- Check `models/hft_ensemble/` for .pkl files

### "SSH tunnel not connected"
- Run: `powershell -File scripts\start_oracle_tunnel.ps1`
- Check: `python scripts/start_hft.py status`

### "IB Gateway not responding"
- SSH to Oracle: `ssh ubuntu@89.168.65.47`
- Check: `docker logs ibgateway`
- Restart: `docker restart ibgateway`

### "TrueFX connection failed"
- Check credentials in `.env`
- Verify network connectivity
- Try alternative source: `source='local'`

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| Accuracy | >60% | >55% |
| AUC | >0.60 | >0.55 |
| Win Rate | >55% | >52% |
| Sharpe | >2.0 | >1.5 |
| Max Drawdown | <5% | <10% |
| Daily PnL | >$50 | >$0 |

## Files to NEVER Modify

- `core/order_book_l3.py` - Core order book logic
- `core/queue_position.py` - Queue models from HftBacktest
- `core/fill_probability.py` - Academic fill models

**If changes needed, create new file and extend.**
