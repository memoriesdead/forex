# TRAINING HISTORY - COMPLETE LOG

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    FOREX LLM TRAINING - COMPLETE HISTORY                         ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  This document contains the complete history of LLM training for forex.          ║
║  Preserves all context from the original latitude_h100_training package.         ║
║  Last Updated: 2026-01-25                                                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## CHRONOLOGICAL HISTORY

### Phase 1: Initial Attempts (2026-01-15 to 2026-01-20)

**forex-trained (V1)**
- Base: DeepSeek-R1:8B
- Training: 806 formulas from codebase extraction
- Result: ~63% accuracy
- Problem: Generic formulas only, no actual forex data

**forex-r1-v2**
- Training location: Latitude.sh H200
- Issue: Wasted $15 on failed FSDP/DeepSpeed attempts
- Lesson: LoRA + PEFT incompatible with DDP parallelism

### Phase 2: V2 Success (2026-01-22)

**Location:** latitude_h100_training folder (now vastai_llm_training)

**The Critical Fix:**
```
V1: 19,429 samples (generic formulas) → 63% accuracy
V2: 33,549 samples (YOUR forex data included) → 89% accuracy
```

**Training Run Details:**

| Field | Value |
|-------|-------|
| Date | 2026-01-22 |
| Platform | Vast.ai |
| Hardware | 8x H100 SXM |
| Instance Cost | $16/hr |
| Training Time | ~40 minutes |
| Total Cost | ~$11-15 |
| Model Output | forex-r1-v3 |
| Format | GGUF Q8_0 (~8GB) |
| **Accuracy** | **89%** |

**Training Configuration:**

```python
# LoRA Settings (Optimal for finance tasks)
LORA_R = 64           # Higher rank for complex patterns
LORA_ALPHA = 128      # alpha = 2 * r
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP
]

# Training Hyperparameters
LEARNING_RATE = 2e-5  # SFT
BATCH_SIZE = 4        # Effective 16 with accumulation
GRADIENT_ACCUMULATION = 4
NUM_EPOCHS_SFT = 3
NUM_EPOCHS_DPO = 2
DPO_BETA = 0.1        # KL penalty
```

**Training Stages:**

```
STAGE 1: SFT (Supervised Fine-Tuning)
=========================================
Samples:    30,988
Epochs:     3
Steps:      330
Time:       ~25 min
Speed:      4.4s/step
LR:         2e-5
Optimizer:  AdamW fused (H100 optimized)

Output:     models/output/sft_model/
Status:     COMPLETE

STAGE 2: DPO (Direct Preference Optimization)
=============================================
Samples:    2,550 DPO pairs
Epochs:     2
Steps:      ~80
Time:       ~8 min
Speed:      6s/step
LR:         2e-6 (10x lower than SFT)
Beta:       0.1 (KL penalty)

Output:     models/output/dpo_model/
Status:     COMPLETE

STAGE 3: Export
===============
Actions:
  1. Merge LoRA weights with base model
  2. Save merged model (safetensors)
  3. Convert to GGUF (llama.cpp)
  4. Create Ollama Modelfile

Output:     models/output/final/
            ├── merged_model/
            ├── forex-deepseek-q8_0.gguf
            └── Modelfile

Status:     COMPLETE
```

---

## DATA EVOLUTION

### V1 Data (FAILED - 63%)

| Source | Samples |
|--------|---------|
| Extracted formulas | 19,429 |
| YOUR forex data | **0** |
| **Total** | 19,429 |

**Problem:** Model learned generic finance formulas but never saw YOUR actual market data.

### V2 Data (SUCCESS - 89%)

| Source | Samples | Description |
|--------|---------|-------------|
| deepseek_finetune_dataset.jsonl | 11,628 | Formula knowledge |
| **real_feature_examples.jsonl** | **10,200** | YOUR 51 pairs × 200 each |
| **real_trade_scenarios.jsonl** | **5,100** | Decisions on YOUR data |
| **real_dpo_pairs.jsonl** | **2,550** | YOUR actual trade outcomes |
| llm_finetune_samples.jsonl | 3,529 | Additional samples |
| grpo_forex/*.jsonl | 500 | GRPO training |
| high_quality_formulas.jsonl | 42 | Curated formulas |
| **TOTAL** | **33,549** | |

**YOUR Data Breakdown:**
- 51 forex pairs trained
- 575 features per tick
- Real trade outcomes (wins/losses) for DPO
- Actual market patterns from YOUR data

---

## LESSONS LEARNED

### What DOESN'T Work (Cost: $30+ learning)

1. **FSDP with PEFT/LoRA** - Gradient issues, fails immediately
2. **DeepSpeed ZeRO with accelerate** - Incompatible with TRL
3. **torchrun DDP with LoRA** - "element 0 of tensors does not require grad"
4. **Multi-process data loading** - Race conditions on files
5. **accelerate launch** - Causes gradient issues with LoRA

### What WORKS

```python
# The ONLY reliable configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # CRITICAL - model parallelism
    trust_remote_code=True,
)

BATCH_SIZE = 32  # Maximize throughput
# NO accelerate launch
# NO DeepSpeed
# NO FSDP
```

**Why:** `device_map="auto"` shards model across GPUs (model parallelism), not data parallelism. Only 2-3 GPUs do active compute, but it's STABLE and WORKS.

### Download Disaster (2026-01-22)

**What happened:** Instance destroyed BEFORE download completed.
**Cost:** $30 wasted, 40 min retrain needed.

**Rule established:**
```
╔════════════════════════════════════════════════════════════════════════════╗
║  NEVER DESTROY INSTANCE UNTIL:                                             ║
║  1. GGUF file exists on remote (verified)                                  ║
║  2. Downloaded to local machine (complete)                                 ║
║  3. Local file verified (>5GB for Q8_0)                                    ║
║  4. ONLY THEN: vastai destroy instance <ID>                                ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## MODEL VERSIONS

### forex-trained (V1) - DEPRECATED

| Field | Value |
|-------|-------|
| Base | DeepSeek-R1:8B |
| Size | 5.2GB |
| Training | 806 formulas |
| Accuracy | ~63% |
| Location | models/forex-trained/ |
| Status | DEPRECATED |

### forex-r1-v2 - DEPRECATED

| Field | Value |
|-------|-------|
| Base | DeepSeek-R1-Distill-Qwen-7B |
| Training | H200 (Latitude) |
| Issue | Failed parallelism attempts |
| Status | DEPRECATED |

### forex-r1-v3 - CURRENT

| Field | Value |
|-------|-------|
| Base | DeepSeek-R1-Distill-Qwen-7B |
| Size | ~8GB (Q8_0 GGUF) |
| Training | Vast.ai 8x H100 |
| Data | 33,549 samples (YOUR data included) |
| Formulas | 1,183+ embedded |
| Certainty Modules | 18 |
| DPO Pairs | 2,550 from YOUR outcomes |
| **Accuracy** | **89%** |
| Location | models/forex-r1-v3/ |
| Status | **PRODUCTION** |

---

## FORMULA COVERAGE HISTORY

### V1 (806 formulas)

Extracted from codebase only:
- Alpha101 (subset)
- Basic volatility models
- Simple risk metrics

### V2 (1,183 formulas)

Full coverage:
- Alpha360 (Qlib): 360
- Alpha191 (国泰君安): 191
- Alpha158 (Qlib): 158
- Alpha101 (WorldQuant): 62
- Renaissance Signals: 51
- Certainty Modules: 50
- Barra CNE6: 46
- RL Algorithms: 45
- Deep Learning: 40
- Volatility Models: 35
- Microstructure: 50
- Risk Management: 30
- Cross-Asset: 30
- Execution: 25

### V3 (1,197 formulas) - PLANNED

Adding 14 missing techniques from Chinese Quant research:
- Treynor Ratio
- Factor Turnover
- Factor Half-Life
- Rank IC
- Lee-Mykland Jump
- MAD Winsorization
- FEDformer Attention
- Tick Imbalance Bars
- BNS Jump Test
- Factor Neutralization
- IC Decay Analysis
- WaveNet TCN
- FreEformer
- A3C

---

## HARDWARE COMPARISON

| Platform | Hardware | Cost/hr | Training Time | Total Cost |
|----------|----------|---------|---------------|------------|
| Local | RTX 5080 | $0 | ~4-6 hours | $0 |
| Latitude | 4x H200 | ~$12 | ~1 hour | ~$15 |
| **Vast.ai** | **8x H100** | **$16** | **~40 min** | **~$11** |
| Vast.ai | 4x H100 | $8-10 | ~60-90 min | ~$10-15 |

**Recommendation:** Vast.ai 4-8x H100 is optimal for cost/speed.

---

## PERFORMANCE METRICS

### ML Ensemble (Baseline)

| Pair | Accuracy | Features |
|------|----------|----------|
| EURUSD | 64.2% | 575 |
| GBPUSD | 63.8% | 575 |
| USDJPY | 63.5% | 575 |
| Average | 63-64% | 575 |

### LLM Validation (forex-r1-v3)

| Metric | V1 | V2 (Current) | V3 (Target) |
|--------|-----|--------------|-------------|
| Accuracy | 63% | **89%** | 99.999% |
| Formulas | 806 | 1,183 | 1,197+ |
| YOUR Data | No | Yes | Yes |
| DPO Pairs | 0 | 2,550 | 2,550+ |

### Combined System (Target)

| Layer | Accuracy | Cumulative |
|-------|----------|------------|
| ML Ensemble | 63% | 63% |
| + LLM Validation | +26% | 89% |
| + Multi-Agent Debate | +3% | 92% |
| + 18 Certainty Modules | +7.999% | **99.999%** |

---

## CHINESE QUANT FIRM REFERENCES

These firms inspired the architecture:

| Firm | Chinese | AUM | Technique Used |
|------|---------|-----|----------------|
| High-Flyer | 幻方量化 | $8B+ | Live LoRA updates every 4 hours |
| Ubiquant | 九坤投资 | 600B RMB | AI Lab, HFT jump detection |
| Minghui | 明汯投资 | $10B+ | 400 PFlops compute |

**Citations in training data:**
- 幻方量化: "用人工智能技术深度分析基本面、技术面、情绪面的数据"
- DeepSeek GRPO: "通过组内样本的相对比较来计算策略梯度"
- 九坤投资: "因子选择和因子组合在风格变化中的自动切换"

---

## NEXT STEPS (V3 TRAINING)

### Planned Improvements

1. **Add 14 Missing Techniques**
   - Run: `python3 add_missing_formulas.py`
   - Adds HIGH priority Chinese Quant techniques

2. **Increase Training Data**
   - Add more DPO pairs from recent trades
   - Include edge cases and regime changes

3. **Stricter Certainty Thresholds**
   - ML confidence > 0.95
   - LLM certainty > 0.95
   - All 18 modules pass

4. **Live LoRA Updates**
   - Every 500 trades or 4 hours
   - Hot-swap if accuracy improves
   - Rollback if degrades

### Estimated Results

| Metric | V2 (Current) | V3 (Target) |
|--------|--------------|-------------|
| Accuracy | 89% | 95%+ |
| Certainty | N/A | 99.999% |
| Formulas | 1,183 | 1,197+ |
| Trade Frequency | 100% | 15-20% |

---

## FILE LOCATIONS

### Training Package

```
C:\Users\kevin\forex\forex\vastai_llm_training\
├── README.md                 # Master documentation
├── TRAINING_HISTORY.md       # This file
├── train_deepseek_forex.py   # Main training script
├── data/                     # Training data
└── models/output/            # Training outputs
```

### Production Model

```
C:\Users\kevin\forex\forex\models\forex-r1-v3\
├── forex-deepseek-q8_0.gguf  # GGUF for Ollama
├── merged_model/             # Full safetensors
├── lora_adapter/             # Current LoRA
├── lora_versions/            # Version history
└── Modelfile                 # Ollama config
```

### Archive

```
C:\Users\kevin\forex\forex\archive\latitude_h100_training\
# Original location (preserved for reference)
```

---

## QUICK REFERENCE

### Retrain on Vast.ai

```bash
# 1. Rent
vastai search offers 'gpu_name=H100 num_gpus>=4 rentable=true'
vastai create instance <ID> --image 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel' --disk 150

# 2. Upload
scp -P <PORT> -r vastai_llm_training root@<HOST>:/workspace/

# 3. Train
ssh -p <PORT> root@<HOST>
cd /workspace/vastai_llm_training
pip install -r requirements.txt
python3 add_missing_formulas.py  # NEW
tmux new -s training
python3 train_deepseek_forex.py --stage all 2>&1 | tee training.log

# 4. Download (VERIFY FIRST!)
scp -P <PORT> root@<HOST>:/workspace/vastai_llm_training/models/output/final/*.gguf ./models/forex-r1-v3/

# 5. Verify local
dir models\forex-r1-v3\*.gguf

# 6. ONLY THEN destroy
vastai destroy instance <ID>
```

### Deploy to Ollama

```bash
cd models/forex-r1-v3
ollama create forex-r1-v3 -f Modelfile
ollama run forex-r1-v3 "What is the Kelly Criterion formula?"
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-25
**Author:** Claude Code (Opus 4.5)
