# VAST.AI/LATITUDE TRAINING - PREFLIGHT CHECKLIST V2

```
+==============================================================================+
|  VERSION 2.0 - INCLUDES YOUR FOREX DATA VERIFICATION                        |
|  CRITICAL FIX: V1 trained on generic formulas only - YOUR data was ignored  |
|  V2: Now verifies YOUR 51 pairs x 15 features are included                  |
|  DATE: 2026-01-22                                                           |
+==============================================================================+
```

---

## PHASE 0: FOREX DATA BRIDGE (NEW IN V2 - DO THIS FIRST!)

### 0.1 Generate Integrated Training Data

**Run LOCALLY before renting any GPU:**

```bash
python scripts/create_llm_forex_bridge.py --verify
python scripts/create_llm_forex_bridge.py --generate
```

### 0.2 Verify Forex Integration

```bash
python scripts/create_llm_forex_bridge.py --verify
```

**Expected output:**
```
[OK] Found 51 forex pairs
[OK] real_feature_examples.jsonl: 10,200 samples
[OK] real_trade_scenarios.jsonl: 5,100 samples
[OK] real_dpo_pairs.jsonl: 2,550 samples
Total integrated samples: 17,850
GRAND TOTAL: 29,478 training samples
[OK] VERIFICATION PASSED
```

### 0.3 Checklist

- [ ] `scripts/create_llm_forex_bridge.py` exists
- [ ] Ran `--generate` successfully
- [ ] `training_data/forex_integrated/` directory exists
- [ ] `real_feature_examples.jsonl` has 10,000+ samples
- [ ] `real_trade_scenarios.jsonl` has 5,000+ samples
- [ ] `real_dpo_pairs.jsonl` has 2,500+ samples
- [ ] **TOTAL samples: 29,000+** (not 19,000 like V1)

**IF ANY OF THESE FAIL: DO NOT PROCEED TO GPU RENTAL**

---

## PHASE 1: LOCAL VERIFICATION

### 1.1 Training Package Files

```bash
python -c "
import os
required = [
    'vastai_h100_training/train_deepseek_forex.py',
    'vastai_h100_training/requirements.txt',
    'training_data/forex_integrated/real_feature_examples.jsonl',  # V2 NEW
    'training_data/forex_integrated/real_trade_scenarios.jsonl',   # V2 NEW
    'training_data/forex_integrated/real_dpo_pairs.jsonl',         # V2 NEW
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print('FAIL:', missing)
else:
    print('PASS: All V2 files exist')
"
```

- [ ] `train_deepseek_forex.py` exists
- [ ] `requirements.txt` exists
- [ ] **V2 NEW:** `forex_integrated/` directory exists with 3 files

### 1.2 Sample Count Verification

```bash
python -c "
import json
from pathlib import Path

# V1 data
v1_count = sum(1 for _ in open('training_data/deepseek_finetune_dataset.jsonl'))
print(f'V1 formula samples: {v1_count:,}')

# V2 data (YOUR FOREX)
v2_dir = Path('training_data/forex_integrated')
v2_count = 0
for f in v2_dir.glob('*.jsonl'):
    v2_count += sum(1 for _ in open(f))
print(f'V2 YOUR forex samples: {v2_count:,}')
print(f'TOTAL: {v1_count + v2_count:,}')

if v2_count < 15000:
    print('FAIL: V2 forex data too small!')
else:
    print('PASS: V2 forex data OK')
"
```

**Expected:**
- V1 formula samples: ~11,628
- V2 YOUR forex samples: ~17,850
- TOTAL: ~29,478

---

## PHASE 2: GPU INSTANCE SELECTION

### 2.1 Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | H100 80GB | 4x H100 NVLink |
| VRAM | 80 GB | 320 GB |
| RAM | 64 GB | 128 GB |
| Disk | 100 GB | 200 GB |

### 2.2 Provider Options

| Provider | Plan | Price/hr | ETA |
|----------|------|----------|-----|
| Latitude.sh | g3.h100.medium (4x H100) | $7.97 | ~45-60 min |
| Vast.ai | H100 80GB | $2.50-4.00 | ~4-6 hr |
| Vast.ai | 2x H100 | ~$5-8 | ~2-3 hr |

### 2.3 Selection Checklist

- [ ] GPU is H100 (not A100, not RTX)
- [ ] Shows 80GB VRAM per GPU
- [ ] On-demand (NOT spot/interruptible)
- [ ] Price noted: $____/hr
- [ ] Estimated total: $____

---

## PHASE 3: UPLOAD DATA

### 3.1 Files to Upload

```bash
# Upload training_data (includes forex_integrated/)
scp -r training_data/ root@<ip>:/workspace/

# Upload training scripts
scp -r vastai_h100_training/ root@<ip>:/workspace/
```

### 3.2 Verify Upload

```bash
# On GPU instance:
ls -la /workspace/training_data/forex_integrated/
# Must show:
# real_feature_examples.jsonl
# real_trade_scenarios.jsonl
# real_dpo_pairs.jsonl

wc -l /workspace/training_data/forex_integrated/*.jsonl
# Must show 17,850+ total lines
```

- [ ] `forex_integrated/` uploaded
- [ ] 3 JSONL files present
- [ ] 17,850+ lines total

---

## PHASE 4: PRE-TRAINING VERIFICATION

### 4.1 Dependencies

```bash
cd /workspace/vastai_h100_training
pip install -r requirements.txt

python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

- [ ] All packages installed
- [ ] CUDA available: True
- [ ] GPU shows H100

### 4.2 Data Load Test

```bash
python3 -c "
import json
from pathlib import Path

# Test V2 forex data loads
forex_dir = Path('/workspace/training_data/forex_integrated')
total = 0
for f in forex_dir.glob('*.jsonl'):
    count = 0
    with open(f) as fh:
        for line in fh:
            json.loads(line)
            count += 1
    print(f'{f.name}: {count}')
    total += count
print(f'Total YOUR forex samples: {total}')
if total < 15000:
    print('FAIL: Not enough forex samples!')
else:
    print('PASS: YOUR forex data ready')
"
```

- [ ] All JSONL files parse correctly
- [ ] 17,850+ samples loaded
- [ ] No encoding errors

---

## PHASE 5: START TRAINING

### 5.1 Create Persistent Session

```bash
tmux new -s training
```

### 5.2 Start Training

```bash
cd /workspace/vastai_h100_training
python3 train_deepseek_forex.py --stage all 2>&1 | tee training.log
```

### 5.3 Verify V2 Data Loading

**Check training log shows:**
```
V2 FOREX INTEGRATION: 15300 SFT + 2550 DPO samples from YOUR data
V2: Added 15300 samples from YOUR forex data
```

**If you see this, YOUR forex data is being used!**

**If you DON'T see "V2 FOREX INTEGRATION", STOP AND FIX!**

- [ ] Training started
- [ ] Log shows "V2 FOREX INTEGRATION"
- [ ] GPU utilization >50%

---

## PHASE 6: MONITORING

### 6.1 Expected Progress

| Stage | Duration | Log Message |
|-------|----------|-------------|
| Data prep | 10 min | "Generated X SFT samples" |
| SFT training | 2-3 hr | "SFT Epoch 1/3" |
| DPO training | 1 hr | "DPO Epoch 1/2" |
| Export | 30 min | "Merging LoRA weights" |

### 6.2 Health Checks (Every 30 min)

```bash
# Check still running
tmux attach -t training

# Check GPU
nvidia-smi

# Check for errors
tail -100 training.log | grep -i error
```

---

## PHASE 7: DOWNLOAD & VERIFY

### 7.1 Download

```bash
# From local machine:
scp root@<ip>:/workspace/vastai_h100_training/models/output/final/*.gguf ./models/
scp root@<ip>:/workspace/vastai_h100_training/models/output/final/Modelfile ./models/
scp root@<ip>:/workspace/vastai_h100_training/training.log ./
```

### 7.2 Verify Training Used YOUR Data

```bash
grep "V2 FOREX INTEGRATION" training.log
# Must show: "V2 FOREX INTEGRATION: 15300 SFT + 2550 DPO samples"

grep "YOUR forex data" training.log
# Must show references to YOUR data
```

- [ ] GGUF file downloaded
- [ ] Log confirms V2 forex data was used
- [ ] File sizes correct

---

## PHASE 8: DEPLOY & TEST

### 8.1 Deploy to Ollama

```bash
cd models
ollama create forex-v2 -f Modelfile
```

### 8.2 Test with YOUR Data Pattern

```bash
ollama run forex-v2 "
EURUSD signal with these features:
  ret_1: 0.0003
  ret_5: 0.0012
  ret_10: 0.0008
  vol_20: 0.00085
  vol_50: 0.00092

ML prediction: BUY (63% confidence)
Should I execute? Use certainty framework.
"
```

**Good response:** References EdgeProof, VPIN, factor alignment, Kelly sizing
**Bad response:** Generic advice without certainty analysis

- [ ] Ollama model created
- [ ] Model responds to test query
- [ ] Response uses certainty framework

---

## PHASE 9: CLEANUP

- [ ] All files downloaded
- [ ] Training log saved
- [ ] **INSTANCE TERMINATED**
- [ ] Total cost: $____

---

## QUICK REFERENCE

### V1 vs V2 Comparison

| Metric | V1 (Old) | V2 (Fixed) |
|--------|----------|------------|
| Formula samples | 19,429 | 11,628 |
| YOUR forex samples | **0** | **17,850** |
| **TOTAL** | 19,429 | **29,478** |
| YOUR data included | NO | YES |

### Critical V2 Checks

1. **Before GPU rental:** Run `create_llm_forex_bridge.py --verify`
2. **After upload:** Verify `forex_integrated/` has 17,850+ samples
3. **During training:** Log must show "V2 FOREX INTEGRATION"
4. **After download:** Verify log confirms YOUR data was used

### Emergency: V2 Data Not Loading

If training log doesn't show "V2 FOREX INTEGRATION":

1. STOP training
2. Check `/workspace/training_data/forex_integrated/` exists
3. If missing, upload from local: `scp -r training_data/forex_integrated/ root@<ip>:/workspace/training_data/`
4. Restart training

---

**Created:** 2026-01-22
**Version:** 2.0
**Critical Fix:** YOUR forex data is now included in training
