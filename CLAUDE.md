# Forex Trading ML Project

```
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT: FOREX - ML + LLM Algorithmic Trading System            ║
║  Path: C:\Users\kevin\forex\forex                                ║
║  TRAINED PAIRS: 51 forex pairs (575 features each)               ║
║  ML ACCURACY: 63-64% (XGBoost + LightGBM + CatBoost)             ║
║  LLM MODEL: forex-r1-v3 (89% accuracy, 8x H100 trained)          ║
║  LIVE TUNING: Chinese Quant Style (幻方/九坤/明汯)                ║
║  TARGET: 80%+ accuracy with continuous learning                  ║
║  ML TRAINING: LOCAL RTX 5080 (16GB VRAM, CUDA 12.9)              ║
║  IB GATEWAY: LOCAL Docker (port 4004, runs 24/7)                 ║
╚══════════════════════════════════════════════════════════════════╝
```

## What This System Does (Plain English)

**A computer program that predicts which way currency prices will move.**

- **51 currency pairs** trained and ready
- **63-64% accuracy** (vs 50% random = 13-14% edge)
- **575 signals** analyzed per prediction
- **3 AI models vote** on each trade (XGBoost, LightGBM, CatBoost)

### Quick Commands

```bash
# Check status
python scripts/train_parallel_max.py --status

# Start paper trading (basic - TrueFX only)
python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD,USDJPY

# FULL COVERAGE - All sources, all symbols, $100 capital (RECOMMENDED)
python scripts/hft_trading_bot.py --mode paper --full-coverage --capital 100

# Or use the batch file
start_full_coverage.bat

# Retrain a pair
python scripts/train_parallel_max.py --symbols EURUSD --no-skip

# LLM MEGA Trading (DeepSeek-R1 Multi-Agent) - NEW!
powershell -File scripts\start_llm_mega.ps1 -Capital 100 -Mode paper

# Or use the batch file
start_llm_trading.bat
```

---

## Custom Forex-Trained LLM (2026-01-20)

**CUSTOM MODEL: forex-trained with 806+ Mathematical Formulas Embedded**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FOREX LLM MEGA TRADING SYSTEM                             ║
║                    CUSTOM forex-trained MODEL ACTIVE                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Custom Model:    forex-trained:latest (806+ formulas embedded)              ║
║  Base Model:      DeepSeek-R1:8B (5.2GB)                                     ║
║  ML Ensemble:     XGBoost + LightGBM + CatBoost (63% base accuracy)          ║
║  LLM Reasoner:    Multi-Agent (Bull/Bear/Risk debate)                        ║
║  Features:        806+ alpha factors (Alpha101 + Alpha191 + Renaissance)     ║
║  Online Learning: Chinese quant style (幻方/九坤/明汯)                        ║
║  Target:          63% → 75%+ win rate with reasoning                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Formula Knowledge (Embedded in forex-trained)

| Category | Formulas | Source |
|----------|----------|--------|
| **Alpha101** | 20+ formulas | WorldQuant (Kakushadze 2016) |
| **Volatility** | HAR-RV, GARCH, EGARCH, GJR-GARCH, Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang | Corsi 2009, Bollerslev 1986 |
| **Microstructure** | VPIN, Kyle Lambda, OFI, Amihud ILLIQ, Roll Spread, Microprice, PIN | Easley 2012, Kyle 1985, Cont 2014 |
| **Risk Management** | Kelly Criterion, VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown | Kelly 1956 |
| **Execution** | Almgren-Chriss, TWAP, VWAP, Avellaneda-Stoikov, Market Impact | Almgren 2001, Avellaneda 2008 |
| **RL Algorithms** | GRPO, PPO, TD, GAE, DQN, SAC | DeepSeek 2025, Schulman 2017 |

**Test the model:**
```bash
ollama run forex-trained "What is Alpha001 formula?"
ollama run forex-trained "Give me the Kelly Criterion formula"
```

### Quick Start

```bash
# PowerShell (full control)
powershell -File scripts\start_llm_mega.ps1 -Capital 100 -Mode paper -LLMMode validation

# Batch file (simple)
start_llm_trading.bat

# Direct Python
python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD,USDJPY --capital 100
```

### LLM Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **advisory** | LLM provides reasoning, doesn't change decisions | Learning/monitoring |
| **validation** | LLM can veto bad ML signals | Production (DEFAULT) |
| **autonomous** | LLM has full control | Experimental only |

### Multi-Agent Architecture

Based on **TradingAgents (UCLA/MIT)** and **幻方量化 (High-Flyer Quant)**:

```
┌─────────────────────────────────────────────────────────────────┐
│                   ML ENSEMBLE (63% accuracy)                     │
│              XGBoost + LightGBM + CatBoost                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               DeepSeek-R1 MULTI-AGENT DEBATE                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ BULL         │  │ BEAR         │  │ RISK         │          │
│  │ Researcher   │  │ Researcher   │  │ Manager      │          │
│  │ (argue FOR)  │  │ (argue       │  │ (size &      │          │
│  │              │  │  AGAINST)    │  │  limits)     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           ▼                                     │
│                 ┌──────────────────┐                            │
│                 │  HEAD TRADER     │                            │
│                 │  (final decision)│                            │
│                 └────────┬─────────┘                            │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
                   EXECUTE or VETO
```

### MCP Server Tools

The LLM trading system exposes tools via MCP at `mcp_servers/llm_trading_mcp.py`:

| Tool | Description |
|------|-------------|
| `llm_analyze_trade` | Full multi-agent analysis with Bull/Bear debate |
| `llm_validate_signal` | Quick yes/no for HFT (<3s) |
| `llm_fast_analyze` | Fast mode analysis (<5s) |
| `llm_explain_regime` | Explain market regime |
| `llm_get_stats` | Decision statistics |
| `llm_set_mode` | Set advisory/validation/autonomous |
| `llm_warmup` | Warmup DeepSeek-R1 model |
| `llm_process_signal` | Full signal processing |

### Core Files

| File | Purpose |
|------|---------|
| `models/forex-trained/Modelfile` | **Custom model definition (806+ formulas)** |
| `core/ml/llm_reasoner.py` | Multi-agent LLM system (uses forex-trained) |
| `mcp_servers/llm_trading_mcp.py` | MCP server for LLM tools |
| `scripts/hft_trading_bot.py` | Trading bot with LLM integration |
| `scripts/start_llm_mega.ps1` | PowerShell launcher |
| `start_llm_trading.bat` | Batch file launcher |

### Training Data (Fine-Tuning)

Extracted from 65k+ lines of quant code:

| Dataset | Samples | Location |
|---------|---------|----------|
| All Formulas | 565 | `training_data/all_formulas.json` |
| Fine-tune Set | 2,711 | `training_data/deepseek_finetune_dataset.jsonl` |
| LLM Samples | 3,529 | `training_data/llm_finetune_samples.jsonl` |

**89.4% of formulas have academic citations!**

### Requirements

- **Ollama** installed at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`
- **forex-trained:latest** model (custom, 5.2GB) - auto-created from Modelfile
- **16GB+ GPU VRAM** (RTX 5080 verified)

```bash
# Check Ollama models
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" list

# Should show:
# forex-trained:latest    5.2 GB    (custom with 806+ formulas)
# deepseek-r1:8b          5.2 GB    (fallback)

# Recreate forex-trained if missing
cd C:\Users\kevin\forex\forex\models\forex-trained
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" create forex-trained -f Modelfile
```

---

## forex-r1-v2: Chinese Quant Live Tuning (2026-01-21)

**GRPO-TRAINED MODEL with LIVE CONTINUOUS LEARNING**

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    FOREX-R1-V2 LIVE TUNING SYSTEM                                ║
║                    Chinese Quant Style (幻方/九坤/明汯)                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Model:           forex-r1-v2 (16GB, GRPO-trained on H200)                       ║
║  Training Data:   100k SFT + 31,890 DPO pairs from real trades                   ║
║  Base Accuracy:   63% (ML ensemble)                                              ║
║  Target:          80%+ with live LLM tuning                                      ║
║  Live Tuning:     Every 4 hours OR 500 trades (RTX 5080)                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### How It Works (Chinese Quant Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LIVE TRADING LOOP                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. Tick arrives                                                               │
│      ↓                                                                          │
│   2. ML Ensemble predicts (63% accuracy)                                        │
│      ↓                                                                          │
│   3. forex-r1-v2 validates with <think>...</think> reasoning                    │
│      ↓                                                                          │
│   4. Execute or veto trade                                                      │
│      ↓                                                                          │
│   5. Record outcome → TradeOutcomeBuffer                                        │
│      ↓                                                                          │
│   6. Every 4 hours: Generate DPO pairs → LoRA fine-tune → Hot-swap              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Chinese Quant Citations (Embedded in Code)

| Source | Technique | Implementation |
|--------|-----------|----------------|
| **幻方量化** | "及时应对市场规则变化，不断更新模型" | `LiveLoRATuner` continuous updates |
| **九坤投资** | "因子选择和因子组合在风格变化中的自动切换" | `DriftDetector` regime detection |
| **DeepSeek GRPO** | "迭代强化学习，历史数据占比10%" | `TradeOutcomeBuffer` 10% replay |
| **BigQuant** | "滚动训练更新预测模型" | Rolling window training |
| **XGBoost增量** | "后期实例决定叶节点权重和新加入的树" | Warm-start LoRA training |
| **模型热更新** | "性能下降时快速回退至历史版本" | Version rollback capability |

### Quick Start

```bash
# 1. Start Live Tuning MCP Server
python mcp_servers/llm_live_tuning_mcp.py --port 8082

# 2. Start Trading (auto-records outcomes for live tuning)
python scripts/hft_trading_bot.py --mode paper --full-coverage --capital 100

# 3. Monitor tuning status
curl http://localhost:8082/api/tuning/status

# 4. Manually trigger training (if 500+ outcomes)
curl -X POST http://localhost:8082/api/tuning/trigger_training
```

### Core Modules

| File | Purpose | Citation |
|------|---------|----------|
| `core/ml/trade_outcome_buffer.py` | DPO pair generation, 10% historical replay | DeepSeek GRPO |
| `core/ml/drift_detector.py` | KS test, KL divergence, HMM regime detection | 九坤投资 |
| `core/ml/live_lora_tuner.py` | Background LoRA training, hot-swap, rollback | 幻方量化 |
| `mcp_servers/llm_live_tuning_mcp.py` | REST API for live tuning | - |
| `models/forex-r1-v2/` | 16GB GRPO-trained model (safetensors) | - |

### Live Tuning API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/tuning/record_outcome` | POST | Record trade outcome for learning |
| `/api/tuning/status` | GET | Get overall tuning status |
| `/api/tuning/drift_status` | GET | Get drift detection status |
| `/api/tuning/trigger_training` | POST | Manually trigger LoRA training |
| `/api/tuning/versions` | GET | List model versions |
| `/api/tuning/rollback` | POST | Rollback to previous version |

### Drift Detection Thresholds

Based on Chinese quant research (九坤投资 2024):

| Metric | Threshold | Action |
|--------|-----------|--------|
| Win rate | < 55% for 2 hours | TRIGGER_RETRAIN |
| Sharpe ratio | < 1.0 for 3 days | MONITOR |
| Drawdown | > 15% | PAUSE_TRADING |
| KS test p-value | < 0.1 | DISTRIBUTION_SHIFT |
| HMM regime change | detected | AGGRESSIVE_REWEIGHT |

### Training Cycle

```
Every 4 hours OR 500 trade outcomes:

1. Export DPO pairs (90% new + 10% historical replay)
   ↓
2. LoRA fine-tune on RTX 5080 (30-60 min)
   - r=64, lr=5e-7, 1 epoch
   ↓
3. Validate on holdout (last 100 trades)
   ↓
4. If accuracy improved ≥1%:
   - Hot-swap LoRA adapter
   - Update Ollama model
   - Trading continues uninterrupted
   ↓
5. Archive buffer, reset drift detector baseline
```

### Model Versions (Rollback Capability)

```bash
# List all versions
curl http://localhost:8082/api/tuning/versions

# Rollback to previous version
curl -X POST http://localhost:8082/api/tuning/rollback \
  -H "Content-Type: application/json" \
  -d '{"version": 3}'
```

### Convert to GGUF for Ollama

```bash
# The model is in safetensors format. To use with Ollama:
python scripts/convert_forex_r1_v2_to_gguf.py

# Or manually with llama.cpp:
python llama-cpp/convert_hf_to_gguf.py models/forex-r1-v2 \
    --outfile models/forex-r1-v2/forex-r1-v2-q8_0.gguf --outtype q8_0

# Then deploy:
cd models/forex-r1-v2
ollama create forex-r1-v2 -f Modelfile
```

### Expected Performance Improvement

| Phase | Accuracy | Method |
|-------|----------|--------|
| ML Ensemble only | 63% | XGBoost + LightGBM + CatBoost |
| + LLM Validation | 70% | forex-r1-v2 veto bad signals |
| + Live Tuning | **80%+** | Continuous DPO from real outcomes |

---

## forex-r1-v3: H100-Trained Live Self-Retraining (2026-01-23)

**PRODUCTION MODEL: Trained on 8x H100 ($15+) with 89% accuracy, live self-retraining**

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    FOREX-R1-V3 LIVE SELF-RETRAINING SYSTEM                       ║
║                    H100-Trained + RTX 5080 Live LoRA Updates                     ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Model:           forex-r1-v3 (8GB GGUF, trained on 8x H100 SXM)                 ║
║  Training Data:   33,549 samples (17,850 YOUR forex data)                        ║
║  Formulas:        1,183 mathematical formulas embedded                           ║
║  Certainty:       18 verification modules                                        ║
║  DPO Pairs:       2,550 from YOUR actual trade outcomes                          ║
║  Base Accuracy:   89% (vs 63% ML ensemble)                                       ║
║  Live Tuning:     Continuous LoRA updates on RTX 5080                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LIVE SELF-RETRAINING SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 1: ML ENSEMBLE (63% accuracy)                                            │
│  └── XGBoost + LightGBM + CatBoost (575 features × 51 pairs)                    │
│                            ↓                                                    │
│  LAYER 2: forex-r1-v3 VALIDATION (89% accuracy)                                 │
│  ├── 1,183 formulas embedded (Alpha101, Kelly, VPIN, etc.)                      │
│  ├── 18 certainty modules (EdgeProof, ICIR, ConformalPrediction)                │
│  └── <think>...</think> reasoning chains                                        │
│                            ↓                                                    │
│  LAYER 3: EXECUTE → RECORD OUTCOME                                              │
│  └── POST /api/tuning/record_outcome (MCP Server :8082)                         │
│                            ↓                                                    │
│  LAYER 4: DRIFT DETECTION (九坤投资 style)                                       │
│  ├── KS test (distribution shift)                                               │
│  ├── KL divergence (prediction drift)                                           │
│  └── HMM regime detection (bull/bear/sideways)                                  │
│                            ↓                                                    │
│  LAYER 5: LIVE LORA TRAINING (幻方量化 style)                                    │
│  ├── Trigger: 500 outcomes OR 4 hours OR drift detected                         │
│  ├── Data: 90% new outcomes + 10% historical replay (DeepSeek GRPO)             │
│  ├── Train: LoRA on RTX 5080 (r=64, alpha=128)                                  │
│  └── Hot-swap if accuracy +1% → Rollback if degraded                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
# 1. Create Ollama model (one-time)
cd models/forex-r1-v3
ollama create forex-r1-v3 -f Modelfile

# 2. Verify model
ollama run forex-r1-v3 "What is the Kelly Criterion formula?"

# 3. Start MCP server for live tuning
python mcp_servers/llm_live_tuning_mcp.py --port 8082

# 4. Start trading with live tuning
python scripts/hft_trading_bot.py --mode paper --full-coverage --capital 100
```

### 18 Certainty Modules

| # | Module | Formula | Purpose |
|---|--------|---------|---------|
| 1 | EdgeProof | H0: μ_strategy ≤ μ_random; p < 0.001 | Statistical edge validation |
| 2 | CertaintyScore | C = w1·IC + w2·ICIR + w3·EdgeProof | Combined confidence |
| 3 | ConformalPrediction | C(x) = {y : s(x,y) ≤ q_{1-α}} | Distribution-free sets |
| 4 | RobustKelly | f* = 0.5 × Kelly × (1-uncertainty) | Safe position sizing |
| 5 | VPIN | Σ\|V_buy - V_sell\| / (n × V_bucket) | Toxicity detection |
| 6 | InformationEdge | IC = corr(predicted, actual) | Signal quality |
| 7 | ICIR | mean(IC) / std(IC) | Consistency ratio |
| 8 | UncertaintyDecomposition | Total = Epistemic + Aleatoric | Uncertainty types |
| 9 | QuantilePrediction | Q_τ(r\|x) for τ ∈ {0.1...0.9} | Return distribution |
| 10 | DoubleAdapt | L = L_task + λ·L_domain | Distribution shift |
| 11 | ModelConfidenceSet | MCS = {m : not rejected at α} | Model comparison |
| 12 | BOCPD | P(r_t\|x_{1:t}) Bayesian changepoint | Regime detection |
| 13 | SHAP | φ_i = Σ Shapley values | Feature attribution |
| 14 | ExecutionCost | IS = (Exec - Decision) × Q | Slippage tracking |
| 15 | AdverseSelection | AS = E[ΔP\|trade] - spread/2 | Informed flow |
| 16 | BayesianSizing | f = ∫ f*(θ)·p(θ\|data)dθ | Posterior sizing |
| 17 | FactorWeighting | w_t = argmax E[r] - λ·Var[r] | Dynamic allocation |
| 18 | RegimeDetection | P(S_t\|X_{1:t}) via HMM | 3-state HMM |

### Files

| File | Purpose |
|------|---------|
| `models/forex-r1-v3/Modelfile` | Ollama model definition (18 modules embedded) |
| `models/forex-r1-v3/forex-deepseek-q8_0.gguf` | 8GB Q8_0 quantized model |
| `models/forex-r1-v3/merged_model/` | Full safetensors for LoRA training |
| `models/forex-r1-v3/lora_adapter/` | Current LoRA adapter |
| `models/forex-r1-v3/lora_versions/` | Version history for rollback |
| `core/ml/trade_outcome_buffer.py` | DPO pair generation |
| `core/ml/drift_detector.py` | KS, KL, HMM detection |
| `core/ml/live_lora_tuner.py` | Background training + hot-swap |
| `mcp_servers/llm_live_tuning_mcp.py` | REST API for tuning |
| `scripts/hft_trading_bot.py` | Trading bot with live tuning |

### MCP Server Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/api/tuning/record_outcome` | POST | Record trade outcome |
| `/api/tuning/status` | GET | Overall tuning status |
| `/api/tuning/drift_status` | GET | Drift detection status |
| `/api/tuning/should_train` | GET | Check if should train |
| `/api/tuning/trigger_training` | POST | Start LoRA training |
| `/api/tuning/versions` | GET | List model versions |
| `/api/tuning/rollback` | POST | Rollback to previous version |
| `/api/tuning/export_dpo` | POST | Export DPO pairs |

### Training Investment Summary

| What You Paid For | How It's Used |
|-------------------|---------------|
| 33,549 training samples | Base model knowledge |
| 1,183 formulas | LLM reasoning with citations |
| 18 certainty modules | Trade validation logic |
| 2,550 DPO pairs | Continued learning from YOUR outcomes |
| H100 training ($15+) | 89% accuracy starting point |
| MCP server | Live API for all operations |
| Drift detection | Auto-trigger retraining |
| Model versioning | Safe rollback if degraded |

### Expected Logs

```
[LIVE_TUNING] Initialized with forex-r1-v3 model
[LIVE_TUNING] MCP server: http://localhost:8082
[LIVE_TUNING] Recorded: EURUSD PnL=$5.20 (buffer: 45, ready: False)
[DRIFT] gradual: Consider increasing retrain frequency
[LIVE_TUNING] Triggering background LoRA training...
```

---

## Full Coverage Trading (2026-01-18)

**Maximum data coverage from ALL sources with $100 capital.**

### Data Sources (Combined)
| Source | Pairs | Type | Cost |
|--------|-------|------|------|
| TrueFX | 10-15 | Free tick streaming | Free |
| IB Gateway | 70+ | Full market data | IB subscription |
| OANDA | 28+ | Streaming quotes | Free API |

**Total unique symbols: 70+**

### Start Full Coverage

```bash
# Option 1: Batch file (simplest)
start_full_coverage.bat

# Option 2: PowerShell with options
powershell -File scripts\start_full_coverage.ps1 -Capital 100 -Mode paper

# Option 3: Direct Python
python scripts/hft_trading_bot.py --mode paper --full-coverage --capital 100 --multi-source

# Option 4: Specific symbols with multi-source
python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD,USDJPY --multi-source --capital 100
```

### Features
- **Best Quote Selection**: Automatically picks lowest spread from all sources
- **Automatic Failover**: If one source fails, others keep working
- **70+ Symbol Coverage**: Trade all available forex pairs
- **Chinese Quant Online Learning**: Models update continuously from live data
- **Kelly Criterion**: 50% Kelly with 50:1 leverage

### Risk Settings (for $100)
| Setting | Value |
|---------|-------|
| Max position | 50% of account |
| Max drawdown | 20% |
| Kelly fraction | 50% (aggressive for small account) |
| Daily trade limit | 100 |
| Leverage | 50:1 |

---

## Trading Instruments (2026-01-23)

**Beyond Spot Forex: Futures, Micro Futures, and Options via IB + CME**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FOREX TRADING INSTRUMENTS AVAILABLE                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTRUMENT          │ LEVERAGE │ MIN MARGIN  │ BEST FOR                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Spot Forex          │ 50:1     │ ~$50        │ Scalping, day trading       ║
║  Micro FX Futures    │ ~25:1    │ ~$200-250   │ Small accounts, swing       ║
║  Currency Futures    │ ~25:1    │ ~$2,500     │ Larger positions            ║
║  FX Options          │ varies   │ ~$100-500   │ Hedging, events             ║
║  Weekly FX Options   │ varies   │ ~$50-200    │ FOMC, NFP plays             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Spot Forex (Current - IDEALPRO)
```python
from ib_insync import Forex
contract = Forex('EURUSD')  # 105+ pairs available
```
- Leverage: 50:1 (US), 30:1 (EU)
- Spreads: ~0.25 pips EUR/USD
- Best for: HFT, scalping, intraday

### Micro FX Futures (CME) - RECOMMENDED FOR $100 CAPITAL
| Symbol | Underlying | Contract Size | Margin |
|--------|------------|---------------|--------|
| M6E | EUR/USD | €12,500 | ~$250 |
| M6B | GBP/USD | £6,250 | ~$200 |
| M6J | USD/JPY | ¥1,250,000 | ~$200 |
| M6A | AUD/USD | A$10,000 | ~$150 |
| M6C | CAD/USD | C$10,000 | ~$150 |

```python
from ib_insync import Future
contract = Future('M6E', exchange='CME')  # Micro Euro
```
- 1/10th size of regular futures
- No rollover/swap costs (vs spot)
- Perfect for small accounts

### Currency Futures (CME)
| Symbol | Underlying | Contract Size | Tick Value |
|--------|------------|---------------|------------|
| 6E | EUR/USD | €125,000 | $12.50 |
| 6B | GBP/USD | £62,500 | $6.25 |
| 6J | USD/JPY | ¥12,500,000 | $12.50 |
| 6A | AUD/USD | A$100,000 | $10.00 |
| 6C | CAD/USD | C$100,000 | $10.00 |
| 6S | CHF/USD | CHF125,000 | $12.50 |

```python
contract = Future('6E', '202503', 'CME')  # March 2025 Euro
```
- Contract months: H (Mar), M (Jun), U (Sep), Z (Dec)
- Higher liquidity than spot for large orders

### FX Options on Futures
```python
from ib_insync import FuturesOption
# EUR/USD 1.10 Call expiring March 21, 2025
contract = FuturesOption('6E', '20250321', 1.10, 'C', 'CME')
```
- European style (exercise at expiry)
- Weekly options for event trading (FOMC, NFP, ECB)
- Defined risk strategies

### When to Use Each

| Scenario | Best Instrument |
|----------|-----------------|
| Scalping (seconds-minutes) | Spot Forex |
| Day trading | Spot Forex or Micro Futures |
| Swing trading (days-weeks) | Micro/Regular Futures |
| Event plays (FOMC, NFP) | Weekly FX Options |
| Hedging positions | FX Options |
| Large positions | Currency Futures |

### IB API Quick Reference
```python
# Spot Forex
forex = Forex('EURUSD')

# Micro Futures
micro = Future('M6E', exchange='CME')

# Regular Futures
future = Future('6E', '202503', 'CME')

# Options on Futures
option = FuturesOption('6E', '20250321', 1.10, 'C', 'CME')

# Get quotes
ib.reqMktData(contract)

# Place order
from ib_insync import MarketOrder
trade = ib.placeOrder(contract, MarketOrder('BUY', 1))
```

---

## ML Training (RTX 5080 - LOCAL)

**GPU:** NVIDIA GeForce RTX 5080 (16GB VRAM)
**CUDA:** 12.9
**Python:** 3.11 with venv at `forex/venv/`

**Train models locally:**
```bash
cd C:\Users\kevin\forex\forex
venv\Scripts\activate
python training_package/train_models.py
```

**Activate venv before any Python work:**
```bash
C:\Users\kevin\forex\forex\venv\Scripts\activate
```

---

## RTX 5080 GPU Maximization (Renaissance-Level)

**Goal:** Max out $3000 RTX 5080 for continuous ML training during live trading.

### Hardware Specs
| Component | Value |
|-----------|-------|
| GPU | RTX 5080 (Blackwell) |
| VRAM | 16GB GDDR7 |
| CUDA Cores | 10,752 |
| Tensor Cores | 336 (5th gen) |
| Bandwidth | 960 GB/s |
| Compute | 12.0 |

### GPU Utilization Targets
| Mode | GPU % | When |
|------|-------|------|
| Idle/Trading | 3-7% | Predictions only |
| Live Retraining | **80-95%** | Every 60s burst |
| Batch Training | **80-95%** | Full retrain |

### Live Retraining System

**Architecture:**
```
Trading Loop (main thread)
    │
    ├── Process tick → Generate features → Predict → Execute
    │
    └── Buffer tick + features → LiveTickBuffer (50k capacity)
                                        │
                                        ↓
                            Background Retrainer (every 60s)
                                        │
                            ┌───────────┴───────────┐
                            │  GPU Training Burst   │
                            │  XGBoost + LightGBM   │
                            │  + CatBoost (GPU)     │
                            │  ~80-95% utilization  │
                            └───────────┬───────────┘
                                        │
                                        ↓
                            Hot-Swap if accuracy +0.5%
```

**Core Files:**
| File | Purpose |
|------|---------|
| `core/gpu_config.py` | GPU-optimized params for XGB/LGB/CB |
| `core/live_tick_buffer.py` | Ring buffer (50k ticks, thread-safe) |
| `core/hot_swappable_ensemble.py` | Atomic model replacement |
| `core/live_retrainer.py` | Background GPU training loop |

### GPU Config Settings (CRITICAL)

```python
# XGBoost - MUST use gpu_hist
'tree_method': 'gpu_hist',    # NOT 'hist' (CPU)
'device': 'cuda',
'max_depth': 12,              # Deep trees for 16GB VRAM
'max_bin': 256,
'predictor': 'gpu_predictor',

# LightGBM
'device': 'gpu',
'max_depth': 12,
'num_leaves': 511,            # 2^12 - 1
'histogram_pool_size': 1024,  # Use GPU memory

# CatBoost
'task_type': 'GPU',           # NOT 'CPU'
'depth': 10,
'border_count': 32,           # GPU optimization
'boosting_type': 'Plain',     # 2-3x faster
```

### Verify GPU Utilization

```powershell
# Monitor GPU during trading
nvidia-smi -l 1

# Should see:
# - 3-7% during normal trading
# - 80-95% spikes every 60 seconds (retraining)
```

### Troubleshooting

**GPU stuck at 7%:**
- Check `tree_method` is `'gpu_hist'` not `'hist'`
- Check `task_type` is `'GPU'` not `'CPU'`
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

**Retraining not starting:**
- Need 1000+ labeled ticks (~10 min of trading)
- Check logs for `[RETRAIN]` messages

---

## IBKR Gateway (SHARED Docker - Stocks + Forex)

### Multi-Market Architecture (2026-01-23)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              SHARED IB GATEWAY - STOCKS & FOREX SIMULTANEOUS                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ONE container, DIFFERENT clientIds = both can trade at same time:           ║
║                                                                              ║
║  ┌─────────────────┐                              ┌─────────────────┐       ║
║  │  FOREX          │     ┌──────────────────┐     │   STOCKS        │       ║
║  │  clientId=100   │────▶│   IB GATEWAY     │◀────│   clientId=5    │       ║
║  │                 │     │   Docker:4004    │     │                 │       ║
║  │                 │     │   DUO423364      │     │                 │       ║
║  └─────────────────┘     │   (paper)        │     └─────────────────┘       ║
║                          │   32 max clients │                               ║
║                          └──────────────────┘                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**SOLUTION:** IB Gateway supports 32 concurrent connections with different clientIds. Forex uses `clientId=100`, Stocks uses `clientId=5`. Both connect to same port 4004.

### Coordination Files (Cross-Project Communication)

| File | Purpose |
|------|---------|
| `C:\Users\kevin\ib_gateway_coordination.json` | JSON coordination status |
| `C:\Users\kevin\IB_GATEWAY_FOR_STOCKS.txt` | Plain text instructions for stocks |

**Stocks Claude should read:** `C:\Users\kevin\IB_GATEWAY_FOR_STOCKS.txt`

### User Credentials

| User | Purpose | Password | Client ID |
|------|---------|----------|-----------|
| `qgzhwj583` | **FOREX** (this PC) | `Jackieismypet12!!` | 100 |
| `stockmarkettrading` | Stocks (other PC) | `Jackieismypet12!!` | 5 |
| `KevinChandarasane` | Main account (don't use directly) | `Jackie12345!!` | N/A |

**Main account `KevinChandarasane` has multiple paper users - always use a specific paper username!**

### ⚠️ CRITICAL: IBKR Pro REQUIRED - Lite Does NOT Work!

```
ERROR: "API support is not available for accounts that support free trading"
CAUSE: Account is IBKR Lite (commission-free but NO API)
FIX:   Upgrade to IBKR Pro at Client Portal → Settings → Account Type
```

**NEVER use IBKR Lite for API/automated trading - it's blocked by IB!**

### Connection Settings

| Setting | FOREX | STOCKS |
|---------|-------|--------|
| **Client ID** | **100** | **5** |
| API Port | 4004 | 4004 (same) |
| Account | DUO423364 | DUO423364 (same) |
| Container | ibgateway | ibgateway (same) |

**Python Connection (FOREX):**
```python
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4004, clientId=100)  # FOREX
```

**Python Connection (STOCKS):**
```python
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4004, clientId=5)  # STOCKS - different clientId!
```

| Gateway Setting | Value |
|-----------------|-------|
| Username | `qgzhwj583` |
| Password | `Jackieismypet12!!` |
| VNC Port | 5900 |
| VNC Password | ibgateway |
| 2FA | Auto-retry, approve on IBKR app weekly |

### Market Hours (CRITICAL)
| Day | Forex Market | IB Gateway |
|-----|--------------|------------|
| **Sunday** | Opens 5 PM ET | 2FA prompt at market open |
| Mon-Thu | Open 24 hours | Connected |
| **Friday** | Closes 5 PM ET | Disconnects at close |
| **Saturday** | CLOSED | "server error" normal |

**Daily Reset:** 12:15-1:45 AM ET (server error normal, auto-retries)

**If you see "** no title **" in logs:** Check if market is open first!

### Quick Commands

```powershell
docker logs ibgateway --tail 10   # Check status
docker restart ibgateway          # Force reconnect
docker ps | findstr ibgateway     # Check if running
```

### Recreate Container (FOREX user)

```powershell
docker rm -f ibgateway
docker run -d --name ibgateway --restart=always -p 4001:4001 -p 4002:4002 -p 4003:4003 -p 4004:4004 -p 5900:5900 -e TWS_USERID=qgzhwj583 -e "TWS_PASSWORD=Jackieismypet12!!" -e TRADING_MODE=paper -e TWS_ACCEPT_INCOMING=yes -e READ_ONLY_API=no -e VNC_SERVER_PASSWORD=ibgateway -e TWOFA_TIMEOUT_ACTION=restart -e RELOGIN_AFTER_TWOFA_TIMEOUT=yes -e EXISTING_SESSION_DETECTED_ACTION=primaryoverride ghcr.io/gnzsnz/ib-gateway:stable
```

### Troubleshooting Concurrent Sessions

**"Client ID already in use" error:**
- Each connection needs unique client ID
- Forex uses `100`, Stocks uses `5`
- Check `.env` → `IB_CLIENT_ID=100`

**"Multiple Paper Trading users" error:**
- Using main account username instead of paper username
- Use `qgzhwj583` (not `KevinChandarasane`)

**Connection kicked out:**
- Another session with same username connected
- Verify correct username for each market

---

## Multi-Broker Trading System (2026-01-18)

**Professional multi-exchange forex trading with intelligent order routing across 5 brokers.**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-BROKER TRADING ARCHITECTURE                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  #  │ BROKER              │ PRIORITY │ STATUS  │ API TYPE                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1  │ Interactive Brokers │    1     │ ENABLED │ IB Gateway (Docker:4004)    ║
║  2  │ OANDA              │    2     │ READY   │ v20 REST API                ║
║  3  │ Forex.com          │    3     │ READY   │ GAIN Capital REST API       ║
║  4  │ tastyfx            │    4     │ READY   │ IG Group REST API           ║
║  5  │ IG Markets         │    5     │ READY   │ IG Group REST API           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Routing Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **BEST_SPREAD** | Route to broker with lowest spread | Default - best execution |
| **PRIORITY** | Route based on broker priority | Prefer specific broker |
| **ROUND_ROBIN** | Distribute orders evenly | Load balancing |
| **FAILOVER** | Primary broker with auto-failover | Reliability focus |

### Quick Start

```bash
# Check broker status
python -c "from config.brokers import print_broker_status; print_broker_status()"

# Start multi-broker trading (IB only by default)
python scripts/multi_broker_trading.py --mode paper --symbols EURUSD,GBPUSD,USDJPY

# Or use batch file
start_multi_broker.bat

# With specific routing strategy
python scripts/multi_broker_trading.py --routing best_spread
python scripts/multi_broker_trading.py --routing failover
```

### Core Files

| File | Purpose |
|------|---------|
| `core/trading/broker_base.py` | Abstract broker interface (~400 lines) |
| `core/trading/broker_ib.py` | Interactive Brokers adapter |
| `core/trading/broker_oanda.py` | OANDA v20 API adapter |
| `core/trading/broker_forex_com.py` | Forex.com adapter |
| `core/trading/broker_tastyfx.py` | tastyfx/IG Markets adapter |
| `core/trading/broker_router.py` | Intelligent order routing (~500 lines) |
| `config/brokers.py` | Broker configuration |
| `scripts/multi_broker_trading.py` | Multi-broker trading bot |

### Adding Broker Credentials

Edit `.env` and add API keys for additional brokers:

```env
# OANDA (https://www.oanda.com/demo-account/)
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=101-001-xxxxxxx-001
OANDA_PAPER=true

# Forex.com (https://developer.forex.com/)
FOREXCOM_USERNAME=your_username
FOREXCOM_PASSWORD=your_password
FOREXCOM_APP_KEY=your_app_key
FOREXCOM_PAPER=true

# tastyfx (https://labs.ig.com/)
TASTYFX_API_KEY=your_api_key
TASTYFX_USERNAME=your_username
TASTYFX_PASSWORD=your_password
TASTYFX_PAPER=true

# IG Markets (https://labs.ig.com/)
IG_API_KEY=your_api_key
IG_USERNAME=your_username
IG_PASSWORD=your_password
IG_PAPER=true
```

### Usage in Code

```python
from core.trading import (
    BrokerRouter,
    RoutingStrategy,
    create_multi_broker_router,
    OrderSide,
    OrderType
)
from config.brokers import load_broker_config_from_env

# Load configs and create router
configs = load_broker_config_from_env()
router = create_multi_broker_router(configs, strategy=RoutingStrategy.BEST_SPREAD)

# Connect to all enabled brokers
router.connect_all()

# Route order to best broker
order = router.route_order(
    symbol='EURUSD',
    side=OrderSide.BUY,
    quantity=10000,
    order_type=OrderType.MARKET
)

# Get quotes from all brokers
quotes = router.get_all_quotes('EURUSD')
best_quote = router.get_best_quote('EURUSD')

# Get aggregate balance across all brokers
balances = router.get_aggregate_balance()
print(f"Total balance: ${balances['total_balance']:,.2f}")

# Cleanup
router.disconnect_all()
```

### Features

- **Best Execution**: Automatically routes to broker with lowest spread
- **Automatic Failover**: If primary broker fails, routes to backup
- **Position Aggregation**: View positions across all brokers
- **Real-time Spread Comparison**: Compare spreads across exchanges
- **Rate Limiting**: Respects broker-specific rate limits
- **Thread-safe**: Safe for concurrent order submission

### Broker Comparison

| Feature | IB | OANDA | Forex.com | tastyfx | IG |
|---------|-----|-------|-----------|---------|-----|
| Min Lot | 25k | 1 | 1k | 0.5 | 0.5 |
| Majors | 14 | 28 | 12 | 14 | 14 |
| API Type | Socket | REST | REST | REST | REST |
| Streaming | Yes | Yes | No | Yes | Yes |

---

## Chinese Quant Online Learning (2026-01-18)

**GOLD STANDARD IMPLEMENTATION - Based on techniques from 幻方量化, 九坤投资, 明汯投资**

**Audit Date:** 2026-01-18 | **Score:** 95% of Chinese Quant Gold Standard

```
╔══════════════════════════════════════════════════════════════════════════╗
║           CHINESE QUANT ONLINE LEARNING SYSTEM (GOLD STANDARD)           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TECHNIQUE          │ IMPLEMENTATION                        │ STATUS     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  增量学习            │ XGBoost/LGB/CB warm-start            │ ✅ DONE    ║
║  热更新              │ process_type='update', refresh_leaf  │ ✅ DONE    ║
║  概念漂移检测         │ KL divergence, KS test              │ ✅ DONE    ║
║  市场状态识别         │ HMM 3-state (牛市/熊市/震荡)         │ ✅ DONE    ║
║  观察反馈            │ 1-tick lookahead labeling            │ ✅ DONE    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  经验回放            │ ReplayBuffer (catastrophic forgetting)│ ✅ NEW    ║
║  定期全量重训练       │ PeriodicFullRetrainer (每10k样本)    │ ✅ NEW    ║
║  堆叠元学习器         │ StackingMetaLearner (XGB meta)       │ ✅ NEW    ║
║  HMM状态检测         │ HMMRegimeDetector (GaussianHMM)      │ ✅ NEW    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Gold Standard Features (Audit 2026-01-18)

| Feature | Purpose | Reference |
|---------|---------|-----------|
| **ReplayBuffer** | Mitigate catastrophic forgetting | [MetaDA Paper](https://arxiv.org/html/2401.03865) |
| **PeriodicFullRetrainer** | "定期全量重训练优于简单增量训练" | [BigQuant](https://bigquant.com/wiki/doc/xKO5e4qSRT) |
| **StackingMetaLearner** | Two-stage ensemble fusion | [ICLR 2025](https://zhuanlan.zhihu.com/p/1903528598360560074) |
| **HMMRegimeDetector** | Proper HMM vs rule-based | [中邮证券 Research](https://finance.sina.com.cn/stock/stockzmt/2025-08-21/doc-infmsyzm6141450.shtml) |

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| **ChineseQuantOnlineLearner** | `core/ml/chinese_online_learning.py` | Main online learning orchestrator |
| **IncrementalXGBoost** | `core/ml/chinese_online_learning.py` | XGBoost warm-start wrapper |
| **IncrementalLightGBM** | `core/ml/chinese_online_learning.py` | LightGBM warm-start wrapper |
| **IncrementalCatBoost** | `core/ml/chinese_online_learning.py` | CatBoost warm-start wrapper |
| **DriftDetector** | `core/ml/chinese_online_learning.py` | KL/KS drift detection |
| **HMMRegimeDetector** | `core/ml/chinese_online_learning.py` | HMM 3-state regime (NEW) |
| **ReplayBuffer** | `core/ml/chinese_online_learning.py` | Catastrophic forgetting (NEW) |
| **PeriodicFullRetrainer** | `core/ml/chinese_online_learning.py` | Rolling full retrain (NEW) |
| **StackingMetaLearner** | `core/ml/chinese_online_learning.py` | Meta-learner ensemble (NEW) |
| **AdaptiveMLEnsemble** | `core/ml/adaptive_ensemble.py` | Trading bot integration |

### Quick Start

```bash
# Start paper trading with online learning (default)
python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD,USDJPY --capital 100

# Or use the batch file
start_trading.bat

# Disable online learning (static models only)
python scripts/hft_trading_bot.py --mode paper --symbols EURUSD --no-online-learning
```

### How It Works

```
Live Tick → Features (575) → Predict → Execute Trade
                ↓
        Store Observation
                ↓
        Next Tick Arrives
                ↓
        Label Previous (actual direction)
                ↓
        Feed to Online Learner
                ↓
        Every 60s: Incremental Update
                ↓
        Hot-swap if accuracy improves
```

### Firm References

| Firm | AUM | Innovation |
|------|-----|------------|
| **幻方量化** (High-Flyer) | $8B+ | Full AI automation since 2017, "萤火" platform |
| **九坤投资** (Ubiquant) | 600B+ RMB | AI Lab, Data Lab, Water Drop Lab |
| **明汯投资** (Minghui) | $10B+ | 400 PFlops AI computing power |

### Official Documentation References

- XGBoost `xgb_model`: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- LightGBM `init_model`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
- CatBoost `init_model`: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier
- hmmlearn: https://hmmlearn.readthedocs.io/en/latest/
- scipy KS test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

---

## Position Management & Take Profit (2026-01-23)

**CRITICAL: Positions must be CLOSED to realize P&L!**

### Take Profit Settings (Auto-Close)

The trading bot automatically closes positions when:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Profit in pips | >= 15 pips | Close immediately |
| Profit percentage | >= 0.3% | Close immediately |
| Time + profit | > 30 min AND profitable | Close (forced) |
| Time + small profit | > 15 min AND > 5 pips | Close (early exit) |

### Manual Position Management

```bash
# Check current P&L and positions
python scripts/check_pnl.py

# Close ALL positions immediately
python scripts/check_pnl.py --close

# Close specific symbol only
python scripts/check_pnl.py --close --symbol EURUSD
```

### Why This Matters

Without closing positions:
- **Unrealized P&L** = paper profit (not real)
- **Realized P&L** = actual profit (in account)
- Positions held forever = no realized gains

---

## Paper Trading Setup (2026-01-18)

**Current Configuration:**

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    $100 PAPER TRADING - READY                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CAPITAL:     $100.00                                                    ║
║  PAIRS:       EURUSD, GBPUSD, USDJPY                                    ║
║  FEATURES:    575 per tick                                               ║
║  MODELS:      XGBoost + LightGBM + CatBoost (9 total)                   ║
║  ACCURACY:    63-65% (13-15% edge over random)                          ║
║  ONLINE:      Chinese Quant 增量学习 enabled                             ║
║  LEVERAGE:    50:1                                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  RISK SETTINGS:                                                          ║
║  - Max Position:    50% per trade                                        ║
║  - Max Drawdown:    20% ($20 loss = stop)                               ║
║  - Kelly Fraction:  50%                                                  ║
║  - Daily Trades:    100 max                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Start Trading:**
```bash
# Option 1: Batch file
start_trading.bat

# Option 2: Command line
python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD,USDJPY --capital 100

# Option 3: Controller script
python scripts/start_hft.py paper --symbols EURUSD,GBPUSD,USDJPY
```

**Market Hours:**
- Opens: Sunday 5:00 PM Eastern (Sydney session)
- Closes: Friday 5:00 PM Eastern
- Best sessions: London (3-4 AM ET), NY (8 AM - 12 PM ET)

---

## MCP Servers (Oracle Cloud)

**Start SSH tunnel for MCP:**
```powershell
powershell -File C:\Users\kevin\forex\forex\scripts\start_oracle_tunnel.ps1
```

**Or manual tunnel:**
```powershell
ssh -i "C:\Users\kevin\forex\forex\ssh-key-89.key" -L 37777:localhost:37777 -L 3847:localhost:3847 -L 8081:localhost:8081 ubuntu@89.168.65.47 -N
```

### Execution Optimization MCP (2026-01-17)

| Setting | Value |
|---------|-------|
| Port | 8081 |
| Script | `scripts/mcp_execution_api.py` |
| Dependencies | Flask, flask-cors |

**Start on Oracle Cloud:**
```bash
cd /home/ubuntu/projects/forex
python scripts/mcp_execution_api.py --port 8081 --host 0.0.0.0
```

**API Endpoints:**
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| POST | `/api/execution/optimize` | Get optimal strategy (MARKET/LIMIT/TWAP/VWAP/AC) |
| POST | `/api/execution/schedule/twap` | Create TWAP schedule |
| POST | `/api/execution/schedule/vwap` | Create VWAP schedule |
| POST | `/api/execution/schedule/ac` | Create Almgren-Chriss schedule |
| POST | `/api/execution/impact` | Estimate market impact |
| GET | `/api/execution/sessions` | Get FX session info |
| GET | `/api/execution/status/<id>` | Get execution status |

**Example - Get Optimal Strategy:**
```bash
curl -X POST http://localhost:8081/api/execution/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "direction": 1,
    "quantity": 1000000,
    "mid_price": 1.0850,
    "spread_bps": 1.0,
    "urgency": 0.5,
    "signal_confidence": 0.7
  }'
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
├── production/             # Production HFT models (canonical)
│   ├── EURUSD_models.pkl
│   ├── GBPUSD_models.pkl
│   └── ...
├── hft_ensemble/           # Alias to production
└── _experimental/          # Experimental models
```

### Risk Management (Built-in)

- **Kelly Criterion**: Fractional Kelly (25%) position sizing
- **Max Drawdown**: 5% hard limit
- **Daily Trade Limit**: 100 trades/day
- **Position Limit**: 2% of account per trade

---

## Modular Multi-Symbol Architecture (2026-01-14)

**Scalable architecture for 78+ forex pairs - replaces monolithic 10,000+ line bot**

### Quick Start

```bash
# Trading
python scripts/trade.py --tier majors --mode paper
python scripts/trade.py --symbols EURUSD,GBPUSD --mode live
python scripts/trade.py --status

# Training
python scripts/train.py --tier majors --parallel 4
python scripts/train.py --symbols EURUSD,GBPUSD,USDJPY
python scripts/train.py --list
```

### Module Structure (Reorganized 2026-01-16)

```
forex/
├── main.py                         # Single entry point
├── config/                         # Configuration
│   └── symbols.py                  # Symbol tiers, pip values
│
├── core/
│   ├── __init__.py                 # Backward-compatible exports
│   │
│   ├── features/                   # Feature engineering
│   │   ├── engine.py               # HFTFeatureEngine
│   │   ├── fast_engine.py          # FastFeatureEngine
│   │   ├── alpha101.py             # Alpha101Complete
│   │   ├── renaissance.py          # RenaissanceSignalGenerator
│   │   └── cross_asset.py          # CrossAssetSignals
│   │
│   ├── data/                       # Data loading
│   │   ├── loader.py               # UnifiedDataLoader
│   │   └── buffer.py               # LiveTickBuffer
│   │
│   ├── ml/                         # ML training
│   │   ├── gpu_config.py           # GPU-optimized params
│   │   ├── parallel_trainer.py     # Pipeline parallel training (NEW)
│   │   ├── ensemble.py             # HotSwappableEnsemble
│   │   ├── retrainer.py            # HybridRetrainer
│   │   └── live_retrainer.py       # LiveRetrainer
│   │
│   ├── execution/                  # Order execution
│   │   ├── order_book.py           # OrderBookL3
│   │   ├── queue_position.py       # Queue models
│   │   ├── fill_probability.py     # Fill estimation
│   │   ├── latency.py              # LatencySimulator
│   │   └── backtest.py             # TickBacktestEngine
│   │
│   ├── models/                     # Model management (existing)
│   ├── risk/                       # Risk management (existing)
│   ├── symbol/                     # Symbol registry (existing)
│   ├── trading/                    # Trading logic (existing)
│   │
│   └── _experimental/              # Non-production modules
│
├── scripts/
│   ├── hft_trading_bot.py          # Main trading bot
│   ├── train_parallel_max.py       # Parallel training CLI (NEW)
│   ├── train_all_mega.py           # Sequential mega training
│   ├── trade.py                    # Trading CLI
│   ├── train.py                    # Training CLI
│   ├── start_hft.py                # Legacy controller
│   ├── toggle_permissions.py       # Permission toggle
│   │
│   ├── trading/                    # Trading scripts (copies)
│   ├── training/                   # Training scripts (copies)
│   ├── data/                       # Data scripts (copies)
│   ├── utils/                      # Utilities (copies)
│   └── _archive/                   # Deprecated scripts (139+)
│
├── models/
│   ├── production/                 # Production models (hft_ensemble)
│   ├── hft_ensemble/               # Alias to production
│   └── _experimental/              # Experimental models
│
└── training_package/               # Self-contained for Vast.ai
```

### Symbol Tiers (51 pairs)

| Tier | Pairs | Examples |
|------|-------|----------|
| majors | 7 | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD |
| crosses | 7 | EURJPY, GBPJPY, EURGBP, EURCHF, AUDJPY, EURAUD, GBPAUD |
| exotics | 7 | EURNZD, GBPNZD, AUDNZD, NZDJPY, AUDCAD, CADCHF, CADJPY |
| eur_crosses | 10 | EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNOK, EURNZD, EURPLN, EURSEK, EURTRY |
| gbp_crosses | 10 | GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNOK, GBPNZD, GBPPLN, GBPSEK, GBPSGD, GBPTRY |
| jpy_crosses | 10 | AUDJPY, CADJPY, CHFJPY, EURJPY, GBPJPY, NZDJPY, SGDJPY, TRYJPY, USDJPY, ZARJPY |

### Key Components

**SymbolRegistry** - Single source of truth:
```python
from core.symbol.registry import SymbolRegistry

registry = SymbolRegistry.get()
majors = registry.get_enabled(tier='majors')  # 7 pairs
all_pairs = registry.get_enabled()            # 51 pairs
config = registry.get_config('EURUSD')        # Per-symbol config
```

**ModelLoader** - LRU cache (700MB vs 5.5GB):
```python
from core.models import ModelLoader

loader = ModelLoader()  # max_cached=10 by default
model = loader.load('EURUSD')  # Loads from disk or cache
available = loader.get_available()  # ['EURUSD', 'GBPUSD', ...]
```

**PortfolioRiskManager** - Per-symbol + portfolio risk:
```python
from core.risk import PortfolioRiskManager

pm = PortfolioRiskManager(total_capital=100000)
can_trade, reason = pm.can_trade('EURUSD', spread_pips=0.5)
size = pm.calculate_position_size('EURUSD', signal_strength=0.8)
```

**TradingBot** - Slim orchestrator (~230 lines):
```python
from core.trading import TradingBot

bot = TradingBot(
    capital=100000,
    tier='majors',  # or symbols=['EURUSD', 'GBPUSD']
    mode='paper'
)
await bot.run()
```

### RAM Optimization

| Before | After |
|--------|-------|
| Load all 78 models = 5.5GB | LRU cache 10 models = 700MB |
| Single risk manager | Per-symbol risk tracking |
| 10,000+ lines monolithic | ~970 lines modular |

### Parallel Training (2026-01-17)

**NEW: Pipeline parallel training for 3x speedup**

```bash
# Check what's trained/remaining
python scripts/train_parallel_max.py --status

# Train all untrained pairs (~1 hour for 51 pairs)
python scripts/train_parallel_max.py --all

# Train specific tier
python scripts/train_parallel_max.py --tier majors

# Train specific symbols (--no-skip to retrain)
python scripts/train_parallel_max.py --symbols EURUSD,GBPUSD --no-skip

# Clear feature cache (free disk space)
python scripts/train_parallel_max.py --clear-cache
```

**Architecture:**
- **Stage 1 (CPU)**: 4 workers generate 575 features in parallel
- **Stage 2 (GPU)**: Train XGBoost + LightGBM + CatBoost
- **Speed**: ~2.6 min/pair (vs 13 min sequential)

**Current Performance (51 pairs trained):**
| Metric | Value |
|--------|-------|
| Pairs Trained | 51 |
| Average Accuracy | 63.48% |
| Best (USDDKK) | 64.70% |
| Features per pair | 575 |

### Legacy Training Scripts

```bash
# Old sequential training (slower)
python scripts/train_all_mega.py --tier majors

# Old parallel (less optimized)
python scripts/train.py --tier majors --parallel 4
```

---

## Chinese Quant Online Learning (2026-01-18)

**Real-time model fine-tuning based on techniques from 幻方量化, 九坤投资, 明汯投资**

### Key Techniques Implemented

| Technique | Chinese Term | Implementation | Source |
|-----------|--------------|----------------|--------|
| Incremental Learning | 增量学习 | `xgb_model` warm start | [XGBoost Docs](https://xgboost.readthedocs.io/) |
| Hot Update | 热更新 | `process_type='update'` | [CSDN](https://blog.csdn.net/xieyan0811/article/details/82949236) |
| Rolling Training | 滚动训练 | Sliding window | [BigQuant](https://bigquant.com/wiki/doc/xVqIPu6RoI) |
| Drift Detection | 概念漂移检测 | KL divergence + KS test | [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) |
| Regime Detection | 市场状态识别 | HMM 3-state | [QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/) |

### Chinese Quant Firm References

| Firm | Chinese | Technique | Source |
|------|---------|-----------|--------|
| High-Flyer | 幻方量化 | Full AI since 2017, "萤火" 10PB platform | [Wikipedia](https://zh.wikipedia.org/zh-hans/幻方量化) |
| Ubiquant | 九坤投资 | AI Lab + Data Lab, 600B RMB AUM | [BigQuant](https://bigquant.com/wiki/doc/R1Y0qSCGV0) |
| Minghui | 明汯投资 | 400P Flops, TOP500 cluster | [Official](https://www.mhfunds.com/) |

### Core Modules

| File | Class | Purpose |
|------|-------|---------|
| `core/ml/chinese_online_learning.py` | `IncrementalXGBoost` | Warm start XGBoost |
| `core/ml/chinese_online_learning.py` | `IncrementalLightGBM` | `init_model` continuation |
| `core/ml/chinese_online_learning.py` | `IncrementalCatBoost` | `init_model` warm start |
| `core/ml/chinese_online_learning.py` | `DriftDetector` | KL/KS drift detection |
| `core/ml/chinese_online_learning.py` | `RegimeDetector` | 3-state market regime |
| `core/ml/chinese_online_learning.py` | `ChineseQuantOnlineLearner` | Complete system |
| `core/ml/adaptive_ensemble.py` | `AdaptiveMLEnsemble` | Trading bot integration |

### Quick Start

```python
from core.ml import AdaptiveMLEnsemble, create_adaptive_ensemble

# Create adaptive ensemble
ensemble = create_adaptive_ensemble()
ensemble.load_models(["EURUSD", "GBPUSD", "USDJPY"])
ensemble.init_online_learning()
ensemble.start_background_updates()

# In trading loop
signal = ensemble.predict(symbol, features)
ensemble.add_observation(symbol, features_array, actual_direction, price)
```

### Benefits vs Old System

| Aspect | Old (retrain from scratch) | New (Chinese Quant) |
|--------|---------------------------|---------------------|
| Update speed | 30-60s | **<1s incremental** |
| Data needed | 50k+ samples | **500 samples** |
| Drift response | Manual | **Auto-detect** |
| Regime adaptation | None | **3-state adaptive** |

### Official Documentation References

- XGBoost: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- LightGBM: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
- CatBoost: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier
- scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html

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
- **Data Storage**: Oracle Cloud (170.9.13.229) - `/home/ubuntu/projects/forex/`
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

Oracle Cloud (170.9.13.229) - /home/ubuntu/projects/forex/
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
- **Runs on**: Oracle Cloud (170.9.13.229)
- **Auto-captures** all coding sessions via hooks
- AI-compressed observations stored in GitHub repository
- Web viewer at Oracle Cloud (tunneled if needed)
- 3-layer search: search() → timeline() → get_observations()
- **Zero manual effort** - captures everything automatically
- **Storage**: GitHub repository (not local)

### 2. Memory-Keeper MCP (Manual Control)
- **Runs on**: Oracle Cloud (170.9.13.229)
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
