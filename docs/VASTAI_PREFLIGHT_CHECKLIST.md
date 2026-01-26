# VAST.AI H100 TRAINING - PREFLIGHT CHECKLIST

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  CRITICAL: DO NOT START TRAINING UNTIL ALL BOXES ARE CHECKED                     ║
║  COST: ~$50-100 for full training run                                            ║
║  RECOVERY: NONE - If training fails, money is lost                               ║
║  CREATED: 2026-01-22                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## PHASE 1: LOCAL VERIFICATION (Before Renting Instance)

### 1.1 Training Package Files

Run this verification script BEFORE uploading:

```bash
python -c "
import os
import sys

required = [
    'vastai_h100_training/train_deepseek_forex.py',
    'vastai_h100_training/requirements.txt',
    'vastai_h100_training/setup_vastai.py',
    'vastai_h100_training/README.md',
]

missing = [f for f in required if not os.path.exists(f)]
if missing:
    print('FAIL: Missing files:', missing)
    sys.exit(1)
print('PASS: All training scripts exist')
"
```

- [ ] `train_deepseek_forex.py` exists and is not empty
- [ ] `requirements.txt` exists with all dependencies
- [ ] `setup_vastai.py` exists for data preparation
- [ ] `README.md` exists with instructions

### 1.2 Training Data Verification

Run this verification script:

```bash
python -c "
import os
import json

print('=== TRAINING DATA CHECK ===')
files = {
    'training_data/all_formulas.json': 2600,  # Min expected
    'training_data/deepseek_finetune_dataset.jsonl': 11000,
    'training_data/llm_finetune_samples.jsonl': 3000,
    'training_data/grpo_forex/train.jsonl': 100,
}

all_ok = True
for path, min_count in files.items():
    if not os.path.exists(path):
        print(f'FAIL: Missing {path}')
        all_ok = False
        continue

    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            count = len(json.load(f))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)

    if count < min_count:
        print(f'FAIL: {path} has {count} items (need {min_count}+)')
        all_ok = False
    else:
        print(f'PASS: {path} = {count:,} items')

if all_ok:
    print('\\n=== ALL DATA CHECKS PASSED ===')
else:
    print('\\n=== DATA CHECKS FAILED - DO NOT PROCEED ===')
"
```

- [ ] `all_formulas.json` has 2,652+ formulas
- [ ] `deepseek_finetune_dataset.jsonl` has 11,628+ samples
- [ ] `llm_finetune_samples.jsonl` has 3,529+ samples
- [ ] `grpo_forex/train.jsonl` has 200+ samples
- [ ] `grpo_forex/val.jsonl` has 50+ samples
- [ ] Total training samples: **19,429+**

### 1.3 Package Size Check

```bash
python -c "
import os

def get_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024*1024)

training_mb = get_size('training_data')
vastai_mb = get_size('vastai_h100_training')
print(f'training_data/: {training_mb:.1f} MB')
print(f'vastai_h100_training/: {vastai_mb:.1f} MB')
print(f'Total upload: {training_mb + vastai_mb:.1f} MB')

if training_mb < 5:
    print('WARNING: training_data seems too small!')
"
```

- [ ] `training_data/` is at least 5 MB
- [ ] `vastai_h100_training/` has all scripts
- [ ] Total upload size is reasonable (<500 MB)

### 1.4 Script Syntax Check

```bash
python -m py_compile vastai_h100_training/train_deepseek_forex.py
python -m py_compile vastai_h100_training/setup_vastai.py
echo "Syntax OK"
```

- [ ] `train_deepseek_forex.py` has no syntax errors
- [ ] `setup_vastai.py` has no syntax errors

---

## PHASE 2: VAST.AI INSTANCE SELECTION

### 2.1 Hardware Requirements

| Requirement | Minimum | Recommended | Status |
|-------------|---------|-------------|--------|
| GPU | H100 80GB | H100 80GB SXM | [ ] |
| VRAM | 80 GB | 80 GB | [ ] |
| RAM | 64 GB | 128 GB | [ ] |
| Disk | 100 GB | 200 GB | [ ] |
| CUDA | 12.0+ | 12.4+ | [ ] |

### 2.2 Instance Selection Checklist

- [ ] **GPU**: H100 (NOT A100, NOT RTX - H100 ONLY)
- [ ] **VRAM**: Shows 80GB available
- [ ] **Reliability**: >95% uptime rating
- [ ] **Location**: US or EU (low latency to HuggingFace)
- [ ] **Price**: Note hourly rate: $____/hr
- [ ] **Estimated cost**: _____ hours × $____/hr = $______

### 2.3 Instance Settings

- [ ] **Docker image**: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` or newer
- [ ] **Disk space**: At least 100GB allocated
- [ ] **On-demand** (NOT spot/interruptible - CRITICAL!)

```
⚠️  NEVER USE SPOT INSTANCES FOR TRAINING
    Spot instances can be terminated mid-training = LOST MONEY
```

---

## PHASE 3: INSTANCE SETUP (After SSH Connected)

### 3.1 Connection Verification

```bash
# Run immediately after SSH connection
nvidia-smi
# Should show H100 with 80GB VRAM

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# Should show: CUDA: True, Device: NVIDIA H100 ...
```

- [ ] SSH connection established
- [ ] `nvidia-smi` shows H100 with 80GB
- [ ] PyTorch sees CUDA device
- [ ] Note instance IP: _______________
- [ ] Note SSH port: _______________

### 3.2 Environment Setup

```bash
# Create working directory
mkdir -p /workspace/forex_training
cd /workspace/forex_training

# Check disk space
df -h /workspace
# Should show 100GB+ available
```

- [ ] Working directory created
- [ ] At least 100GB disk space available

### 3.3 Upload Files (DO NOT SKIP)

```bash
# FROM YOUR LOCAL MACHINE (not the instance):
scp -P <port> -r training_data/ root@<ip>:/workspace/forex_training/
scp -P <port> -r vastai_h100_training/ root@<ip>:/workspace/forex_training/
```

- [ ] `training_data/` uploaded successfully
- [ ] `vastai_h100_training/` uploaded successfully
- [ ] Verify with `ls -la /workspace/forex_training/`

### 3.4 Verify Upload Integrity

```bash
# ON THE INSTANCE:
cd /workspace/forex_training

# Count files
find training_data -name "*.jsonl" | wc -l
# Should be 9+ files

find training_data -name "*.json" | wc -l
# Should be 1+ files

# Check main training file
wc -l training_data/deepseek_finetune_dataset.jsonl
# Should show 11628+ lines
```

- [ ] All JSONL files present (9+)
- [ ] `all_formulas.json` present
- [ ] Line counts match local counts

### 3.5 Install Dependencies

```bash
cd /workspace/forex_training/vastai_h100_training

# Install requirements
pip install -r requirements.txt

# Verify critical packages
python3 -c "
import torch
import transformers
import peft
import trl
print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'trl: {trl.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

- [ ] All pip packages installed without errors
- [ ] torch version 2.2.0+
- [ ] transformers version 4.40.0+
- [ ] peft version 0.10.0+
- [ ] trl version 0.8.0+
- [ ] CUDA available: True
- [ ] GPU shows H100

---

## PHASE 4: PRE-TRAINING VERIFICATION

### 4.1 Model Download Test

```bash
# Test that we can download the model (DO THIS FIRST)
python3 -c "
from transformers import AutoTokenizer
print('Downloading tokenizer (tests HuggingFace access)...')
tok = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
print('SUCCESS: HuggingFace access works')
print(f'Vocab size: {tok.vocab_size}')
"
```

- [ ] Tokenizer downloads successfully
- [ ] No authentication errors
- [ ] No network errors

### 4.2 GPU Memory Test

```bash
python3 -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Free VRAM: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserved')

# Allocate test tensor
x = torch.randn(10000, 10000, device='cuda')
print(f'Test allocation successful')
del x
torch.cuda.empty_cache()
"
```

- [ ] Shows ~80GB total VRAM
- [ ] Test allocation works
- [ ] No OOM errors

### 4.3 Training Data Load Test

```bash
cd /workspace/forex_training
python3 -c "
import json
from pathlib import Path

# Test loading all_formulas.json
with open('training_data/all_formulas.json', 'r') as f:
    formulas = json.load(f)
print(f'Loaded {len(formulas)} formulas')

# Test loading JSONL
count = 0
with open('training_data/deepseek_finetune_dataset.jsonl', 'r') as f:
    for line in f:
        json.loads(line)
        count += 1
print(f'Loaded {count} JSONL samples')
print('SUCCESS: All data loads correctly')
"
```

- [ ] `all_formulas.json` loads (2,652 formulas)
- [ ] JSONL files parse correctly
- [ ] No encoding errors

### 4.4 Dry Run Test

```bash
cd /workspace/forex_training/vastai_h100_training

# Run with --dry-run to test without actual training
python3 train_deepseek_forex.py --stage prepare --dry-run 2>&1 | head -50
```

- [ ] Script starts without import errors
- [ ] Data preparation logic runs
- [ ] No path errors

---

## PHASE 5: START TRAINING

### 5.1 Create Persistent Session

```bash
# Use tmux or screen so training continues if SSH disconnects
tmux new -s training
# OR
screen -S training
```

- [ ] tmux/screen session created
- [ ] Session name: `training`

### 5.2 Start Training with Logging

```bash
cd /workspace/forex_training/vastai_h100_training

# Start training with full logging
python3 train_deepseek_forex.py --stage all 2>&1 | tee training.log
```

- [ ] Training started
- [ ] Log file being written to `training.log`
- [ ] Note start time: _______________

### 5.3 Detach and Monitor

```bash
# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t training

# Monitor GPU usage (in separate terminal)
watch -n 5 nvidia-smi
```

- [ ] Detached from tmux (training continues)
- [ ] GPU utilization >50%
- [ ] No errors in log

---

## PHASE 6: DURING TRAINING MONITORING

### 6.1 Health Checks (Run Every 30 Minutes)

```bash
# Check training still running
tmux attach -t training
# Look for progress output, then Ctrl+B, D to detach

# Check GPU
nvidia-smi

# Check disk space
df -h /workspace

# Check for errors in log
tail -100 /workspace/forex_training/vastai_h100_training/training.log | grep -i "error\|exception\|failed"
```

- [ ] Training process still running
- [ ] GPU utilization normal (>50%)
- [ ] Disk space available (>20GB free)
- [ ] No error messages in log

### 6.2 Expected Progress Milestones

| Stage | Expected Duration | Checkpoint |
|-------|-------------------|------------|
| Model download | 5-10 min | [ ] Complete |
| SFT data prep | 5-10 min | [ ] Complete |
| SFT training | 2-3 hours | [ ] Complete |
| DPO data prep | 5-10 min | [ ] Complete |
| DPO training | 1-2 hours | [ ] Complete |
| LoRA merge | 15-30 min | [ ] Complete |
| GGUF export | 15-30 min | [ ] Complete |
| **TOTAL** | **4-6 hours** | [ ] **DONE** |

---

## PHASE 7: POST-TRAINING VERIFICATION

### 7.1 Output Files Check

```bash
ls -la /workspace/forex_training/vastai_h100_training/models/output/

# Should see:
# - sft_model/ (LoRA adapter after SFT)
# - dpo_model/ (LoRA adapter after DPO)
# - final/merged_model/ (Full merged model)
# - final/forex-deepseek.gguf (Quantized for Ollama)
# - final/Modelfile (Ollama config)
```

- [ ] `sft_model/` directory exists
- [ ] `dpo_model/` directory exists
- [ ] `final/merged_model/` directory exists
- [ ] `forex-deepseek.gguf` file exists
- [ ] `Modelfile` exists

### 7.2 Model Size Verification

```bash
# Check GGUF file size (should be 4-8GB for Q8_0)
ls -lh /workspace/forex_training/vastai_h100_training/models/output/final/*.gguf

# Check merged model size (should be ~14-15GB)
du -sh /workspace/forex_training/vastai_h100_training/models/output/final/merged_model/
```

- [ ] GGUF file is 4-8 GB
- [ ] Merged model is 14-15 GB
- [ ] Files are not truncated/corrupted

### 7.3 Quick Inference Test

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/workspace/forex_training/vastai_h100_training/models/output/final/merged_model'
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

prompt = 'What is the Kelly Criterion formula?'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Response: {response}')
print('SUCCESS: Model generates text')
"
```

- [ ] Model loads without errors
- [ ] Model generates coherent response
- [ ] Response mentions Kelly Criterion correctly

---

## PHASE 8: DOWNLOAD RESULTS

### 8.1 Download to Local Machine

```bash
# FROM YOUR LOCAL MACHINE:
mkdir -p downloaded_model

# Download GGUF (most important)
scp -P <port> root@<ip>:/workspace/forex_training/vastai_h100_training/models/output/final/forex-deepseek.gguf downloaded_model/

# Download Modelfile
scp -P <port> root@<ip>:/workspace/forex_training/vastai_h100_training/models/output/final/Modelfile downloaded_model/

# Download full model (optional, large)
scp -P <port> -r root@<ip>:/workspace/forex_training/vastai_h100_training/models/output/final/merged_model/ downloaded_model/

# Download training log
scp -P <port> root@<ip>:/workspace/forex_training/vastai_h100_training/training.log downloaded_model/
```

- [ ] GGUF file downloaded
- [ ] Modelfile downloaded
- [ ] Training log downloaded
- [ ] (Optional) Full merged model downloaded

### 8.2 Local Verification

```bash
# Check downloaded file sizes
ls -lh downloaded_model/

# Verify GGUF not corrupted
file downloaded_model/forex-deepseek.gguf
# Should show: data or GGUF format identifier
```

- [ ] GGUF file is correct size (matches instance)
- [ ] File not corrupted

### 8.3 Deploy to Ollama

```bash
cd downloaded_model
ollama create forex-deepseek -f Modelfile

# Test
ollama run forex-deepseek "What is Alpha001 formula?"
```

- [ ] Ollama model created successfully
- [ ] Model responds to test query
- [ ] Response contains formula knowledge

---

## PHASE 9: INSTANCE CLEANUP

### 9.1 Stop and Delete Instance

- [ ] All files downloaded and verified locally
- [ ] Training log saved
- [ ] **STOP BILLING**: Instance terminated on Vast.ai
- [ ] Note total hours used: _____ hours
- [ ] Note total cost: $_____

---

## EMERGENCY PROCEDURES

### If SSH Disconnects During Training

1. **DON'T PANIC** - Training continues in tmux/screen
2. Reconnect: `ssh -p <port> root@<ip>`
3. Reattach: `tmux attach -t training`
4. Check log: `tail -100 training.log`

### If Training Crashes

1. Check error: `tail -500 training.log`
2. Check GPU memory: `nvidia-smi`
3. If OOM: Reduce batch size in script and restart
4. If disk full: `df -h` and clear cache: `rm -rf ~/.cache/huggingface/hub/*`

### If Instance Becomes Unresponsive

1. Try SSH again after 5 minutes
2. Check Vast.ai dashboard for instance status
3. If truly dead, instance may have been reclaimed (rare for on-demand)
4. **This is why we use ON-DEMAND, not spot instances**

### If Download Interrupted

1. Resume with `rsync`:
   ```bash
   rsync -avz --progress -e "ssh -p <port>" root@<ip>:/workspace/forex_training/vastai_h100_training/models/output/final/ downloaded_model/
   ```
2. Verify file sizes match

---

## FINAL SIGN-OFF

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  TRAINING COMPLETION CERTIFICATE                                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  Date: ____________________                                                      ║
║                                                                                  ║
║  Instance ID: ____________________                                               ║
║                                                                                  ║
║  Total Training Time: _______ hours                                              ║
║                                                                                  ║
║  Total Cost: $________                                                           ║
║                                                                                  ║
║  Model Downloaded: [ ] Yes  [ ] No                                               ║
║                                                                                  ║
║  Ollama Deployed: [ ] Yes  [ ] No                                                ║
║                                                                                  ║
║  Test Passed: [ ] Yes  [ ] No                                                    ║
║                                                                                  ║
║  Instance Terminated: [ ] Yes  [ ] No                                            ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

**REMEMBER:**
1. **NEVER use spot instances** - they can be terminated
2. **ALWAYS use tmux/screen** - survives SSH disconnect
3. **DOWNLOAD BEFORE TERMINATING** - no recovery after
4. **VERIFY DOWNLOADS LOCALLY** - check file sizes match

**Created:** 2026-01-22
**For:** Vast.ai H100 DeepSeek Forex Training
**Formulas:** 1,172+ unique (2,652 training samples)
**Target:** 63% → 99.999% certainty
