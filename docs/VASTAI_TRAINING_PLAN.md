# VAST.AI TRAINING PLAN: 63% → 99.999% CERTAINTY

```
╔════════════════════════════════════════════════════════════════════════════════╗
║  PROJECT: FOREX GOLD STANDARD ML MODEL                                         ║
║  GOAL: 63% Accuracy → 99.999999% CERTAINTY                                     ║
║  METHOD: Combine CHINA + USA + ACADEMIC Gold Standards                         ║
║  COMPUTE: Vast.ai H100 (80GB VRAM)                                             ║
║  TIMELINE: 12-48 hours training                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1: Data Preparation](#phase-1-data-preparation)
3. [Phase 2: Base Model Selection](#phase-2-base-model-selection)
4. [Phase 3: Formula Integration](#phase-3-formula-integration)
5. [Phase 4: Training Pipeline](#phase-4-training-pipeline)
6. [Phase 5: Certainty Modules](#phase-5-certainty-modules)
7. [Phase 6: Validation & Deployment](#phase-6-validation--deployment)
8. [Model Sources & Citations](#model-sources--citations)

---

## 1. ARCHITECTURE OVERVIEW

### The Gold Standard Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FOREX GOLD STANDARD MODEL                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 5: CERTAINTY VALIDATION (18 Modules)                         │   │
│  │  EdgeProof, ConformalPrediction, BOCPD, SHAP, Kelly...              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 4: LLM REASONING (Multi-Agent Debate)                        │   │
│  │  Bull Researcher | Bear Researcher | Risk Manager | Head Trader     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 3: ML ENSEMBLE (GPU-Accelerated)                             │   │
│  │  XGBoost + LightGBM + CatBoost (63% base accuracy)                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 2: FEATURE ENGINEERING (1,172+ Formulas / 2,652 Training)    │   │
│  │  Alpha360 + Alpha158 + Alpha191 + Alpha101 + Barra + Renaissance    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 1: DATA SOURCES                                              │   │
│  │  51 Forex Pairs | 19,429 Formula Samples | 562,835 Training Ticks   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Target Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| **Accuracy** | 63% | 70%+ | ML Ensemble + LLM Validation |
| **Certainty** | Unknown | 99.999% | 18 Statistical Modules |
| **Edge Proof** | Unvalidated | p < 0.001 | Bootstrap Hypothesis Testing |
| **Win Rate** | ~60% | 65%+ | Kelly-Optimal Sizing |

---

## PHASE 1: DATA PREPARATION

### Checklist

- [ ] **1.1 Verify Training Data Exists**
  ```bash
  # Check training_package/ has all 51 pairs
  ls training_package/*/train.parquet | wc -l  # Should be 51
  ```

- [ ] **1.2 Verify Formula Datasets**
  | Dataset | Samples | Status |
  |---------|---------|--------|
  | high_quality_formulas.jsonl | 31 | [ ] |
  | deepseek_finetune_dataset.jsonl | 11,628 | [ ] |
  | fingpt_forex_formulas.jsonl | 12 | [ ] |
  | llm_finetune_samples.jsonl | 3,529 | [ ] |
  | llm_finetune_conversations.jsonl | 200 | [ ] |
  | llm_finetune_dataset.jsonl | 3,529 | [ ] |
  | grpo_forex/train.jsonl | 200 | [ ] |
  | grpo_forex/val.jsonl | 50 | [ ] |
  | grpo_forex/all_formulas.jsonl | 250 | [ ] |
  | all_formulas.json | 2,652 | [ ] |
  | **TOTAL** | **19,429+ samples** | |

- [ ] **1.3 Generate Additional Training Data**
  ```bash
  # Generate certainty module examples
  python scripts/generate_comprehensive_training_data.py

  # Generate DPO pairs from trade outcomes
  python scripts/prepare_grpo_training.py
  ```

- [ ] **1.4 Validate Formula Coverage**
  | Category | Target | Source | Status |
  |----------|--------|--------|--------|
  | **Alpha360** (Microsoft Qlib) | 360 factors | `core/features/alpha360.py` | [ ] |
  | **Alpha158** (Microsoft Qlib) | 158 factors | `core/features/alpha158.py` | [ ] |
  | **Alpha191** (国泰君安) | 191 factors | `core/features/gitee_chinese_factors.py` | [ ] |
  | **Alpha101** (WorldQuant) | 62 factors | `core/features/alpha101.py` | [ ] |
  | **Barra CNE6** (MSCI adapted) | 46 factors | `core/features/barra_cne6.py` | [ ] |
  | **Renaissance Signals** | 50 signals | `core/features/renaissance.py` | [ ] |
  | **Volatility** (HAR-RV, GARCH, Yang-Zhang) | 25+ models | `core/features/academic_volatility.py` | [ ] |
  | **Microstructure** (VPIN, OFI, Kyle) | 30+ factors | `core/features/chinese_gold_standard.py` | [ ] |
  | **Risk** (Kelly, Sharpe, VaR, Sortino) | 20+ formulas | `core/_experimental/quant_formulas.py` | [ ] |
  | **Execution** (Almgren-Chriss, TWAP, AS) | 15+ algorithms | `core/execution/` | [ ] |
  | **RL Algorithms** (GRPO, PPO, TD, SAC) | 35+ algorithms | `core/rl/` | [ ] |
  | **Deep Learning** (iTransformer, TFT) | 15+ models | `core/features/academic_deep_learning.py` | [ ] |
  | **Cross-Asset Signals** | 17+ signals | `core/features/cross_asset.py` | [ ] |
  | **Universal Math** (OU, Cointegration) | 30+ formulas | `core/features/universal_math.py` | [ ] |
  | **Japan/Korea/India/Europe Quant** | 100+ factors | `core/features/*_quant.py` | [ ] |
  | **Certainty Modules** | 18 modules | Training script embedded | [ ] |
  | **TOTAL** | **1,172+ unique formulas** | all_formulas.json: 2,652 | |

---

## PHASE 2: BASE MODEL SELECTION

### Option A: Chinese Quant Origin (RECOMMENDED)

| Model | Size | Why | Citation |
|-------|------|-----|----------|
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | Built BY $8B hedge fund (幻方量化) | [DeepSeek 2025](https://github.com/deepseek-ai/DeepSeek-R1) |
| XuanYuan-70B | 70B | #1 on FinEval, first Chinese 100B finance | [XuanYuan](https://huggingface.co/Duxiaoman-DI/XuanYuan-70B) |
| Qwen3-8B | 8B | +22% return in live trading test | [Qwen](https://huggingface.co/Qwen) |

### Option B: USA Finance Models

| Model | Size | Why | Citation |
|-------|------|-----|----------|
| **FinGPT v3.3** | 13B | Beats GPT-4 on sentiment, $300 to train | [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) |
| AdaptLLM/finance-LLM | 13B | Competes with BloombergGPT-50B | [HuggingFace](https://huggingface.co/AdaptLLM/finance-LLM) |
| Llama-3-SEC-70B | 70B | SEC filings specialist | [HuggingFace](https://huggingface.co/blog/Crystalcareai/llama-3-sec) |
| Plutus-Llama-3.1-8B | 8B | 394 finance books embedded | [HuggingFace](https://huggingface.co/0xroyce/Plutus-Meta-Llama-3.1-8B-Instruct-bnb-4bit) |

### Option C: Hybrid (GOLD STANDARD)

```
Base: DeepSeek-R1-Distill-Qwen-7B (Chinese hedge fund)
  + FinGPT sentiment capabilities (USA)
  + Alpha101 + Alpha191 formulas (WorldQuant + 国泰君安)
  + FinRL position sizing (USA)
  + 18 Certainty Modules (Custom)
```

### Checklist

- [ ] **2.1 Select Base Model**
  - [ ] Option A: DeepSeek-R1-Distill-Qwen-7B
  - [ ] Option B: FinGPT v3.3
  - [ ] Option C: Hybrid approach

- [ ] **2.2 Download Base Model**
  ```bash
  # On Vast.ai H100
  huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  ```

- [ ] **2.3 Verify VRAM Requirements**
  | Model | Full Precision | 4-bit Quantized |
  |-------|---------------|-----------------|
  | 7B | 14 GB | 4 GB |
  | 13B | 26 GB | 8 GB |
  | 70B | 140 GB | 40 GB |

  **H100 has 80GB - can run 70B quantized or 7B-13B full precision**

---

## PHASE 3: FORMULA INTEGRATION

### 3.1 Chinese Quant Formulas

- [ ] **Alpha191 (国泰君安 Guotai Junan)**
  ```
  Citation: 国泰君安证券. "Alpha191因子库"
  Source: Gitee Chinese Quant repositories
  Formulas: 191 alpha factors for A-share market
  ```

- [ ] **幻方量化 (High-Flyer Quant) Techniques**
  ```
  Citation: 幻方量化. "用人工智能技术深度分析基本面、技术面、情绪面的数据"
  Source: https://blog.csdn.net/zk168_net/article/details/108076246
  Techniques: Online learning, model hot-update, regime detection
  ```

- [ ] **九坤投资 (Ubiquant) Techniques**
  ```
  Citation: 九坤投资. "因子选择和因子组合在风格变化中的自动切换"
  Source: BigQuant research papers
  Techniques: Adaptive factor weighting, drift detection
  ```

### 3.2 USA Academic Formulas

- [ ] **Alpha101 (WorldQuant)**
  ```
  Citation: Kakushadze, Z. (2016). "101 Formulaic Alphas". SSRN
  Source: https://arxiv.org/abs/1601.00991
  Formulas: 101 cross-sectional alpha factors
  ```

- [ ] **HAR-RV Volatility**
  ```
  Citation: Corsi, F. (2009). "A Simple Approximate Long-Memory Model of
            Realized Volatility". Journal of Financial Econometrics
  Formula: RV_{t+1} = β₀ + β_d·RV_d + β_w·RV_w + β_m·RV_m
  ```

- [ ] **VPIN (Informed Trading)**
  ```
  Citation: Easley, D., López de Prado, M., O'Hara, M. (2012).
            "Flow Toxicity and Liquidity in a High-Frequency World"
  Formula: VPIN = Σ|V_buy - V_sell| / (n × V_bucket)
  ```

- [ ] **Kelly Criterion**
  ```
  Citation: Kelly, J.L. (1956). "A New Interpretation of Information Rate".
            Bell System Technical Journal
  Formula: f* = (b·p - q) / b
  ```

- [ ] **Almgren-Chriss Execution**
  ```
  Citation: Almgren, R. & Chriss, N. (2001). "Optimal Execution of
            Portfolio Transactions". Journal of Risk
  Formula: n_j = n_0·sinh(κ(T-t))/sinh(κT)
  ```

### 3.3 Reinforcement Learning

- [ ] **GRPO (DeepSeek)**
  ```
  Citation: DeepSeek (2025). "Group Relative Policy Optimization"
  Source: https://github.com/deepseek-ai/DeepSeek-R1
  Formula: L = E[min(r·A, clip(r)·A) - β·KL]
  Key: Eliminates critic network, uses group-relative rewards
  ```

- [ ] **PPO (Schulman)**
  ```
  Citation: Schulman, J. et al. (2017). "Proximal Policy Optimization"
  Source: https://arxiv.org/abs/1707.06347
  Formula: L = E[min(r·A, clip(r, 1±ε)·A)]
  ```

- [ ] **FinRL Framework**
  ```
  Citation: AI4Finance Foundation (2020). "FinRL: Financial Reinforcement Learning"
  Source: https://github.com/AI4Finance-Foundation/FinRL (13.7k+ stars)
  Agents: PPO, A2C, DDPG, TD3, SAC
  ```

### Checklist

- [ ] **3.1 Verify all formula files exist**
- [ ] **3.2 Generate training samples for each formula**
- [ ] **3.3 Add academic citations to all samples**
- [ ] **3.4 Create formula → trading scenario mappings**

---

## PHASE 4: TRAINING PIPELINE

### 4.1 Stage 1: Supervised Fine-Tuning (SFT)

```
Purpose: Teach model the 1,172+ mathematical formulas
Data: 19,429 formula Q&A samples
Method: LoRA fine-tuning (efficient)
Time: 2-4 hours on H100
```

**Checklist:**

- [ ] **4.1.1 Configure LoRA Parameters**
  ```python
  # Optimal settings from FinGPT research
  LORA_R = 64           # Rank (higher for more capacity)
  LORA_ALPHA = 128      # Alpha = 2 * r
  LORA_DROPOUT = 0.05
  TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
  ```

- [ ] **4.1.2 Set Training Hyperparameters**
  ```python
  LEARNING_RATE = 2e-5
  BATCH_SIZE = 4        # H100 can handle larger batches
  GRADIENT_ACCUMULATION = 4  # Effective batch = 16
  NUM_EPOCHS = 3
  WARMUP_RATIO = 0.05
  ```

- [ ] **4.1.3 Run SFT Training**
  ```bash
  python train_deepseek_forex.py --stage sft
  ```

- [ ] **4.1.4 Validate Formula Knowledge**
  ```bash
  # Test model knows formulas
  ollama run forex-model "What is Alpha001 formula?"
  ollama run forex-model "Explain Kelly Criterion"
  ollama run forex-model "Calculate VPIN"
  ```

### 4.2 Stage 2: DPO/GRPO Training

```
Purpose: Learn from trade outcomes (chosen vs rejected)
Data: DPO pairs from actual trade results
Method: GRPO-style preference learning
Time: 1-2 hours on H100
```

**Checklist:**

- [ ] **4.2.1 Generate DPO Pairs**
  | Type | Example |
  |------|---------|
  | Chosen | "VETO - VPIN 0.82 too high, wait for normalization" |
  | Rejected | "APPROVE - ML model confident so trade anyway" |

- [ ] **4.2.2 Configure DPO Parameters**
  ```python
  DPO_BETA = 0.1        # KL penalty strength
  DPO_LOSS_TYPE = "sigmoid"
  LEARNING_RATE = 2e-6  # Lower than SFT
  NUM_EPOCHS = 2
  ```

- [ ] **4.2.3 Run DPO Training**
  ```bash
  python train_deepseek_forex.py --stage dpo
  ```

### 4.3 Stage 3: Export & Deploy

- [ ] **4.3.1 Merge LoRA Weights**
  ```bash
  python train_deepseek_forex.py --stage export
  ```

- [ ] **4.3.2 Convert to GGUF**
  ```bash
  python llama.cpp/convert_hf_to_gguf.py models/output/merged \
      --outfile forex-gold-standard.gguf --outtype q8_0
  ```

- [ ] **4.3.3 Create Ollama Model**
  ```bash
  ollama create forex-gold-standard -f Modelfile
  ```

- [ ] **4.3.4 Download to Local Machine**
  ```bash
  scp -P <port> root@<vastai_ip>:/workspace/forex-gold-standard.gguf ./models/
  ```

---

## PHASE 5: CERTAINTY MODULES

### The 18 Certainty Modules

```
CERTAINTY ≠ ACCURACY

Accuracy = Win rate (63% → 70%+)
Certainty = Statistical confidence the edge is REAL (99.999%)

Renaissance doesn't win 100% of trades.
They are 100% CERTAIN their edge exists.
```

### Tier 1: Core Certainty (Must Have)

- [ ] **EdgeProof** - Bootstrap hypothesis testing
  ```
  H0: μ_strategy ≤ μ_random
  Reject if p-value < 0.001
  Citation: Efron, B. (1979). "Bootstrap Methods"
  ```

- [ ] **CertaintyScore** - Multi-factor confidence
  ```
  C = w1·IC + w2·ICIR + w3·EdgeProof + w4·Conformal
  ```

- [ ] **ConformalPrediction** - Distribution-free intervals
  ```
  C(x) = {y : s(x,y) ≤ q_{1-α}(S_cal)}
  Citation: Vovk et al. (2005). "Algorithmic Learning in a Random World"
  ```

- [ ] **RobustKelly** - Half-Kelly with uncertainty
  ```
  f* = 0.5 × Kelly × (1 - uncertainty)
  ```

- [ ] **VPIN** - Informed trading detection
  ```
  VPIN > 0.5 → High toxicity → VETO trade
  Citation: Easley et al. (2012)
  ```

### Tier 2: Advanced Certainty

- [ ] **InformationEdge** - IC measurement
- [ ] **ICIR** - IC Information Ratio
- [ ] **UncertaintyDecomposition** - Epistemic vs Aleatoric
- [ ] **QuantilePrediction** - Full distribution
- [ ] **DoubleAdapt** - Domain adaptation

### Tier 3: Expert Certainty

- [ ] **ModelConfidenceSet** - Best model identification
- [ ] **BOCPD** - Changepoint detection
- [ ] **SHAP** - Feature importance
- [ ] **ExecutionCost** - Implementation shortfall
- [ ] **AdverseSelection** - Informed trader detection
- [ ] **BayesianSizing** - Parameter uncertainty
- [ ] **FactorWeighting** - Dynamic optimization
- [ ] **RegimeDetection** - HMM market states

### Checklist

- [ ] **5.1 Verify all 18 modules implemented**
  ```python
  from core.ml import (
      EdgeProof, CertaintyScore, ConformalPrediction,
      RobustKelly, VPIN, InformationEdge, ICIR,
      UncertaintyDecomposition, QuantilePrediction, DoubleAdapt,
      ModelConfidenceSet, BOCPD, SHAP, ExecutionCost,
      AdverseSelection, BayesianSizing, FactorWeighting, RegimeDetection
  )
  ```

- [ ] **5.2 Add certainty examples to training data**
- [ ] **5.3 Train model on certainty reasoning**
- [ ] **5.4 Validate 99.999% certainty threshold**

---

## PHASE 6: VALIDATION & DEPLOYMENT

### 6.1 Model Validation

- [ ] **6.1.1 Formula Knowledge Test**
  | Test | Pass Criteria |
  |------|---------------|
  | Alpha001 explanation | Correct formula + citation |
  | Kelly Criterion | Correct formula + example |
  | VPIN calculation | Correct formula + interpretation |
  | HAR-RV model | All 3 components explained |

- [ ] **6.1.2 Trading Scenario Test**
  | Scenario | Expected Output |
  |----------|-----------------|
  | High VPIN (0.8+) | VETO with explanation |
  | Low confidence (52%) | VETO - no edge |
  | Good signal + OFI confirm | APPROVE with Kelly sizing |
  | News event | VETO - wait 30 min |

- [ ] **6.1.3 Certainty Module Test**
  | Module | Pass Criteria |
  |--------|---------------|
  | EdgeProof | p-value < 0.001 |
  | Conformal | Coverage ≥ 95% |
  | VPIN | Correctly identifies toxic flow |

### 6.2 Integration Testing

- [ ] **6.2.1 Paper Trading (24 hours minimum)**
  ```bash
  python scripts/hft_trading_bot.py --mode paper --full-coverage --capital 100
  ```

- [ ] **6.2.2 Metrics to Track**
  | Metric | Target |
  |--------|--------|
  | Win Rate | > 60% |
  | Sharpe Ratio | > 2.0 |
  | Max Drawdown | < 10% |
  | VETO Rate | 30-50% (rejecting bad signals) |

- [ ] **6.2.3 Compare vs Baseline**
  | System | Expected Accuracy |
  |--------|-------------------|
  | ML Only | 63% |
  | ML + LLM Validation | 70%+ |
  | ML + LLM + Certainty | 99.999% certainty |

### 6.3 Deployment

- [ ] **6.3.1 Deploy to Local Ollama**
  ```bash
  cd models/forex-gold-standard
  ollama create forex-gold-standard -f Modelfile
  ollama run forex-gold-standard "Test query"
  ```

- [ ] **6.3.2 Update Trading Bot Config**
  ```python
  # In scripts/hft_trading_bot.py
  LLM_MODEL = "forex-gold-standard"
  LLM_MODE = "validation"  # LLM can veto bad signals
  ```

- [ ] **6.3.3 Enable Live Tuning**
  ```bash
  python mcp_servers/llm_live_tuning_mcp.py --port 8082
  ```

---

## MODEL SOURCES & CITATIONS

### Chinese Quant Models

| Model/Source | Citation | Link |
|--------------|----------|------|
| **DeepSeek-R1** | DeepSeek (2025). "DeepSeek-R1: Incentivizing Reasoning" | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |
| **XuanYuan 轩辕** | Du Xiaoman Finance. "XuanYuan: First Chinese 100B Financial LLM" | [HuggingFace](https://huggingface.co/Duxiaoman-DI/XuanYuan-70B) |
| **Alpha191** | 国泰君安证券. "Alpha191因子库" | Gitee |
| **幻方量化** | High-Flyer Quant. "萤火" AI Platform | [Wikipedia](https://zh.wikipedia.org/wiki/幻方量化) |
| **九坤投资** | Ubiquant. "AI Lab + Data Lab" | [BigQuant](https://bigquant.com) |

### USA Models

| Model/Source | Citation | Link |
|--------------|----------|------|
| **FinGPT** | Yang et al. (2023). "FinGPT: Open-Source Financial LLMs" | [GitHub](https://github.com/AI4Finance-Foundation/FinGPT) |
| **FinRL** | Liu et al. (2020). "FinRL: Financial Reinforcement Learning" | [GitHub](https://github.com/AI4Finance-Foundation/FinRL) |
| **AdaptLLM** | Cheng et al. (2024). "Adapting LLMs to Domains" | [HuggingFace](https://huggingface.co/AdaptLLM/finance-LLM) |
| **Llama-3** | Meta (2024). "Llama 3" | [HuggingFace](https://huggingface.co/meta-llama) |

### Academic Formulas

| Formula | Citation |
|---------|----------|
| **Alpha101** | Kakushadze, Z. (2016). "101 Formulaic Alphas". SSRN |
| **HAR-RV** | Corsi, F. (2009). J. Financial Econometrics |
| **VPIN** | Easley, López de Prado, O'Hara (2012). Review of Financial Studies |
| **Kelly** | Kelly, J.L. (1956). Bell System Technical Journal |
| **Almgren-Chriss** | Almgren & Chriss (2001). Journal of Risk |
| **GARCH** | Bollerslev, T. (1986). J. Econometrics |
| **PPO** | Schulman et al. (2017). arXiv:1707.06347 |
| **GRPO** | DeepSeek (2025). arXiv |

### RL Frameworks

| Framework | Stars | Citation |
|-----------|-------|----------|
| **FinRL** | 13.7k+ | AI4Finance Foundation (2020) |
| **Stable-Baselines3** | 12k+ | Raffin et al. (2021) |
| **QLib** | 35k+ | Microsoft Research Asia (2020) |

---

## QUICK START COMMANDS

```bash
# ============================================
# VAST.AI H100 TRAINING - QUICK START
# ============================================

# 1. Upload training package
scp forex_deepseek_training_*.tar.gz root@<vastai_ip>:/workspace/

# 2. SSH into instance
ssh -p <port> root@<vastai_ip>

# 3. Extract and setup
tar -xzf forex_deepseek_training_*.tar.gz
cd vastai_training
pip install -r requirements.txt

# 4. Prepare data
python setup_vastai.py --prepare --validate

# 5. Run full training (SFT + DPO + Export)
python train_deepseek_forex.py --stage all

# 6. Download trained model
exit
scp -P <port> root@<vastai_ip>:/workspace/vastai_training/models/output/final/* ./models/

# 7. Deploy locally
cd models
ollama create forex-gold-standard -f Modelfile
ollama run forex-gold-standard "What is Alpha001?"

# 8. Start trading
python scripts/hft_trading_bot.py --mode paper --full-coverage --capital 100
```

---

## SUCCESS CRITERIA

| Phase | Criteria | Status |
|-------|----------|--------|
| **Data** | 19,429+ samples prepared | [ ] |
| **Base Model** | DeepSeek-R1-7B loaded | [ ] |
| **SFT** | Loss < 1.0, formula knowledge verified | [ ] |
| **DPO** | Correct APPROVE/VETO behavior | [ ] |
| **Export** | GGUF model created | [ ] |
| **Deploy** | Ollama model running | [ ] |
| **Validate** | Win rate > 60%, VETO rate 30-50% | [ ] |
| **Certainty** | EdgeProof p < 0.001 | [ ] |

---

## FINAL CHECKLIST

- [ ] All training data prepared (19,429+ samples)
- [ ] Base model selected and downloaded
- [ ] SFT training completed (1,172+ formulas learned)
- [ ] DPO training completed (trade reasoning learned)
- [ ] Model exported to GGUF
- [ ] Model deployed to local Ollama
- [ ] Paper trading validated (24+ hours)
- [ ] Certainty modules integrated
- [ ] EdgeProof shows p < 0.001
- [ ] Ready for live trading

---

**Created:** 2026-01-22
**Author:** Claude Code
**Project:** FOREX Gold Standard ML Model
**Goal:** 63% Accuracy → 99.999999% Certainty
