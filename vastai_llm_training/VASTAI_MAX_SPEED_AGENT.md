# Vast.ai MAX SPEED Training Agent

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VAST.AI SPEED OPTIMIZATION - LESSONS LEARNED                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Package:        vastai_llm_training (renamed from latitude_h100_training)       ║
║  Optimal Config: device_map="auto" + BATCH_SIZE=32                               ║
║  Training Time:  ~40 min on 8x H100 SXM                                          ║
║  Cost:           ~$11 at $16/hr                                                  ║
║  Achieved:       89% accuracy                                                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## CRITICAL: DOWNLOAD BEFORE DESTROY

```
╔════════════════════════════════════════════════════════════════════════════╗
║  ⚠️⚠️⚠️  NEVER DESTROY INSTANCE UNTIL DOWNLOAD VERIFIED  ⚠️⚠️⚠️                ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  When Vast.ai instance is DESTROYED, ALL DATA IS PERMANENTLY LOST.         ║
║  There is NO recovery. You MUST retrain from scratch (~$15 + 40 min).      ║
║                                                                            ║
║  CORRECT ORDER:                                                            ║
║  1. Training completes → GGUF created                                      ║
║  2. Download GGUF to local machine                                         ║
║  3. VERIFY file exists locally (>5GB for Q8_0)                             ║
║  4. ONLY THEN: vastai destroy instance <ID>                                ║
║                                                                            ║
║  WRONG ORDER (NEVER DO):                                                   ║
║  1. Training completes                                                     ║
║  2. vastai destroy instance  ← ALL DATA LOST!                              ║
║  3. Try to download ← TOO LATE!                                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## CRITICAL LEARNINGS (Burned $30 learning these)

### What DOESN'T Work:
1. **FSDP with PEFT/LoRA** - Gradient issues, fails immediately
2. **DeepSpeed ZeRO with accelerate** - Compatibility issues with TRL
3. **torchrun DDP with LoRA** - "element 0 of tensors does not require grad" error
4. **Multi-process data loading race conditions** - Files not ready on all ranks

### What WORKS:
- **Original script with `device_map="auto"` + HIGH BATCH SIZE**
- Model shards across GPUs (model parallelism)
- Only 2-3 GPUs do active compute, but it's STABLE
- Batch size 32 = ~7 samples/sec throughput

## Optimal Configuration

```python
# In train_deepseek_forex.py
BATCH_SIZE = 32  # Max for 8x H100

# Model loading - KEEP device_map="auto"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # DO NOT REMOVE
    trust_remote_code=True,
)
```

## Quick Start Commands

```bash
# 1. Rent 8x H100 SXM on Vast.ai ($16/hr)
vastai search offers 'gpu_name=H100 num_gpus>=8 rentable=true'
vastai create instance <ID> --image 'pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel' --disk 150

# 2. Upload training package
scp -P <PORT> -r vastai_llm_training root@ssh5.vast.ai:/workspace/

# 3. Install deps
ssh -p <PORT> root@ssh5.vast.ai
cd /workspace/vastai_llm_training
pip install -r requirements.txt

# 4. Set batch size to 32
sed -i 's/BATCH_SIZE = [0-9]*/BATCH_SIZE = 32/' train_deepseek_forex.py

# 5. Run in tmux
tmux new -s train
python3 train_deepseek_forex.py --stage all 2>&1 | tee training.log

# 6. Monitor
nvidia-smi -l 5
tail -f training.log

# 7. DESTROY when done to stop billing
vastai destroy instance <ID>
```

## Expected Performance

| Stage | Steps | Time | Speed |
|-------|-------|------|-------|
| SFT | 330 | ~25 min | 4.4s/step |
| DPO | ~80 | ~8 min | 6s/step |
| GGUF | - | ~5 min | - |
| **TOTAL** | - | **~40 min** | - |

## Download Results

```bash
# From Windows
scp -P <PORT> root@ssh5.vast.ai:/workspace/vastai_llm_training/models/output/final/*.gguf C:\Users\kevin\forex\forex\models\forex-r1-v3\
```

## Cost Estimate

- 8x H100 SXM: $16/hr
- Training time: ~40 min
- **Total: ~$11**

## Why Not True 8x Parallelism?

The 7B model with LoRA CAN fit on single H100, but:
- PEFT + DDP = gradient issues
- FSDP + LoRA = incompatible
- DeepSpeed + TRL = version conflicts

The `device_map="auto"` approach:
- Shards model weights across GPUs
- Only 2-3 GPUs do compute
- Still 2.5x faster than single GPU
- MOST IMPORTANTLY: IT WORKS

## Checkpoints

Checkpoints save every 100 steps. If training dies:
```bash
# Resume automatically - script finds latest checkpoint
python3 train_deepseek_forex.py --stage all
```

## DOWNLOAD PROCEDURE (MANDATORY - READ BEFORE DESTROY)

### Step 1: Verify Training Complete
```bash
# Check for GGUF file
ssh -p <PORT> root@<HOST> "ls -la /workspace/vastai_llm_training/models/output/final/*.gguf"
# Must show file >5GB
```

### Step 2: Download (WAIT FOR COMPLETION)
```bash
# From Windows PowerShell
scp -P <PORT> root@<HOST>:/workspace/vastai_llm_training/models/output/final/*.gguf C:\Users\kevin\forex\forex\models\forex-r1-v3\
```

### Step 3: Verify Local File
```powershell
dir "C:\Users\kevin\forex\forex\models\forex-r1-v3\*.gguf"
# MUST show file with size >5GB
```

### Step 4: ONLY NOW Destroy
```bash
vastai destroy instance <ID>
```

## Safe Download Script

Use the automated script that verifies before destroy:
```powershell
.\scripts\download_vastai_model.ps1 -InstanceId <ID> -Port <PORT> -SshHost <HOST>
```
