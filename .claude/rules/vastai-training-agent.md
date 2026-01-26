# Vast.ai Training Agent Rules (CRITICAL - FOLLOW EXACTLY)

## MANDATORY: Read These Files First

Before ANY Vast.ai training operation, READ these files in order:

1. `vastai_llm_training/README.md` - Quick start guide
2. `vastai_llm_training/VASTAI_MAX_SPEED_AGENT.md` - Speed optimization (CRITICAL)
3. `vastai_llm_training/GOLD_STANDARD_VERIFICATION.md` - Verification checklist
4. `.claude/rules/vastai-safety.md` - Safety rules

## NEVER VIOLATE THESE RULES

### 1. DOWNLOAD BEFORE DESTROY (CRITICAL)

```
╔════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  NEVER DESTROY INSTANCE UNTIL DOWNLOAD IS VERIFIED LOCALLY  ⚠️         ║
║                                                                            ║
║  When instance is destroyed, ALL DATA IS PERMANENTLY LOST.                 ║
║  There is NO recovery. You MUST retrain from scratch (~$15 + 40 min).      ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**Correct order:**
1. Training completes → GGUF file created
2. Download GGUF to local machine
3. VERIFY file exists locally with correct size (>5GB for Q8_0)
4. ONLY THEN destroy instance

**Wrong order (NEVER DO):**
1. Training completes
2. Destroy instance  ← WRONG! Model lost forever!
3. Try to download ← TOO LATE!

### 2. Use VASTAI_MAX_SPEED_AGENT.md Config

**What WORKS (use this):**
```python
device_map="auto"      # Model shards across GPUs
BATCH_SIZE = 32        # Max throughput
# NO accelerate launch # Causes gradient issues with LoRA
```

**What DOESN'T work (never use):**
- `accelerate launch` with PEFT/LoRA
- DeepSpeed ZeRO with TRL
- `device_map=None` with multi-GPU
- FSDP with LoRA

### 3. Verify V2 FOREX INTEGRATION

Training log MUST show:
```
V2 FOREX INTEGRATION: 15300 SFT + 2550 DPO samples from YOUR data
```

If this message is NOT present, STOP and fix before continuing.

### 4. Monitor Checkpoints

Checkpoints save every 100 steps. Verify they exist:
```bash
ls -la /workspace/vastai_llm_training/models/output/sft/checkpoint-*
```

### 5. Expected Timeline (8x H100)

| Stage | Steps | Time | Speed |
|-------|-------|------|-------|
| SFT | 330 | ~25 min | 4.4s/step |
| DPO | ~80 | ~8 min | 6s/step |
| GGUF | - | ~5 min | - |
| **TOTAL** | - | **~40 min** | - |

## Download Procedure (MANDATORY)

### Step 1: Verify Training Complete

```bash
# Check for completion message
grep -E "(COMPLETE|GGUF|Saved)" /workspace/vastai_llm_training/training.log | tail -5

# Verify GGUF file exists
ssh -p <PORT> root@<HOST> "ls -la /workspace/vastai_llm_training/models/output/final/*.gguf"
```

### Step 2: Download Model

```bash
# From Windows - WAIT for this to complete!
scp -P <PORT> root@<HOST>:/workspace/vastai_llm_training/models/output/final/*.gguf C:\Users\kevin\forex\forex\models\forex-r1-v3\
```

### Step 3: Verify Local File

```powershell
# Check file exists and size is correct
dir "C:\Users\kevin\forex\forex\models\forex-r1-v3\*.gguf"
# Must show file >5GB for Q8_0, >3GB for Q4_K_M
```

### Step 4: ONLY NOW Destroy Instance

```bash
# After verification ONLY
vastai destroy instance <ID>
```

## Automated Download Script

Use `scripts/download_vastai_model.ps1` for safe downloads:

```powershell
.\scripts\download_vastai_model.ps1 -InstanceId <ID> -Port <PORT> -SshHost <HOST>
```

This script:
1. Checks remote file exists
2. Downloads with progress
3. Verifies local file size
4. Asks for confirmation before destroy

## If Training Dies

Checkpoints save every 100 steps. To resume:

```bash
# Training auto-resumes from latest checkpoint
python3 train_deepseek_forex.py --stage all
```

## Cost Reference

| Action | Cost |
|--------|------|
| 8x H100 SXM | $16/hr |
| Full training (~40 min) | ~$11 |
| Premature destroy (retrain) | ~$11 WASTED |
| Extra 5 min to verify download | ~$1.50 |

**ALWAYS spend the $1.50 to verify.**

## Quick Reference Commands

```bash
# Check training progress
ssh -p <PORT> root@<HOST> "tail -5 /workspace/vastai_llm_training/training.log | grep -E '(%|/330|loss)'"

# Check GPU utilization
ssh -p <PORT> root@<HOST> "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# Check if GGUF exists
ssh -p <PORT> root@<HOST> "ls -la /workspace/vastai_llm_training/models/output/final/*.gguf"

# Download (WAIT for completion)
scp -P <PORT> root@<HOST>:/workspace/vastai_llm_training/models/output/final/*.gguf ./models/forex-r1-v3/

# Verify local
dir models\forex-r1-v3\*.gguf

# ONLY THEN destroy
vastai destroy instance <ID>
```
