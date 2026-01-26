# VAST.AI MASTER CHECKLIST (READ EVERY TIME)

```
╔════════════════════════════════════════════════════════════════════════════════╗
║  ⚠️⚠️⚠️  ABSOLUTE RULE: DOWNLOAD VERIFIED BEFORE DESTROY  ⚠️⚠️⚠️                  ║
║                                                                                ║
║  On 2026-01-22, training was LOST because instance was destroyed BEFORE       ║
║  download. Cost: $30+ wasted, 40 min retrain. NEVER AGAIN.                     ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

## Current Instance Info

| Setting | Value |
|---------|-------|
| **Instance ID** | 30393763 |
| **SSH** | `ssh -p 33762 root@ssh6.vast.ai` |
| **SCP** | `scp -P 33762 root@ssh6.vast.ai:/path ./local` |
| **Cost** | $16/hr |
| **Download Path** | `/workspace/vastai_llm_training/models/output/final/*.gguf` |
| **Local Path** | `C:\Users\kevin\forex\forex\models\forex-r1-v3\` |

## MANDATORY FILES TO READ

Before ANY Vast.ai operation:
1. [ ] `vastai_llm_training/README.md`
2. [ ] `vastai_llm_training/VASTAI_MAX_SPEED_AGENT.md`
3. [ ] `vastai_llm_training/GOLD_STANDARD_VERIFICATION.md`
4. [ ] `.claude/rules/vastai-safety.md`
5. [ ] `.claude/rules/vastai-training-agent.md`
6. [ ] **THIS FILE** `.claude/rules/00-vastai-master-checklist.md`

---

## PRE-TRAINING CHECKLIST

- [ ] Instance rented with 8x H100
- [ ] Training package uploaded
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] BATCH_SIZE=32 in script
- [ ] device_map="auto" (NOT accelerate launch)
- [ ] Training started in tmux session
- [ ] V2 FOREX INTEGRATION logged (your 51 pairs data)

## DURING TRAINING CHECKLIST

- [ ] Check progress: `grep -E '(%|/330|/40|loss)' training.log | tail -5`
- [ ] Checkpoints saving every 100 steps
- [ ] No errors in log

## POST-TRAINING CHECKLIST (BEFORE DESTROY)

### Step 1: Verify GGUF Exists on Remote
```bash
ssh -p 33762 root@ssh6.vast.ai "ls -la /workspace/vastai_llm_training/models/output/final/*.gguf"
```
- [ ] File exists
- [ ] File size > 5GB (for Q8_0)

### Step 2: Download to Local Machine
```bash
# From PowerShell - WAIT for completion
scp -P 33762 root@ssh6.vast.ai:/workspace/vastai_llm_training/models/output/final/*.gguf C:\Users\kevin\forex\forex\models\forex-r1-v3\
```
- [ ] Command completed (no errors)
- [ ] Progress showed file transfer

### Step 3: VERIFY Local File Exists
```powershell
dir "C:\Users\kevin\forex\forex\models\forex-r1-v3\*.gguf"
```
- [ ] File listed
- [ ] Size > 5GB (not 0 bytes, not partial)

### Step 4: Test File is Valid (Optional but Recommended)
```powershell
# Try to load first few bytes
python -c "open(r'C:\Users\kevin\forex\forex\models\forex-r1-v3\forex-r1-v3-Q8_0.gguf', 'rb').read(1024)"
```
- [ ] No errors

### Step 5: ONLY NOW Destroy Instance
```bash
vastai destroy instance 30393763
```
- [ ] Downloaded verified ✅
- [ ] Local file exists ✅
- [ ] Size correct ✅
- [ ] NOW safe to destroy

---

## SAFE DOWNLOAD SCRIPT (USE THIS!)

```powershell
.\scripts\download_vastai_model.ps1 -InstanceId 30393763 -Port 33762 -SshHost ssh6.vast.ai
```

This script:
1. ✅ Checks remote file exists
2. ✅ Downloads with progress
3. ✅ Verifies local file size
4. ✅ Asks for confirmation before destroy

---

## DANGER ZONE - WHAT DESTROYS YOUR MODEL

| Action | Result |
|--------|--------|
| `vastai destroy instance <ID>` | ALL DATA PERMANENTLY LOST |
| Instance timeout/expiry | ALL DATA PERMANENTLY LOST |
| SSH disconnect | Safe (tmux keeps running) |
| Power outage on local | Safe (remote keeps running) |

---

## COST COMPARISON

| Choice | Cost |
|--------|------|
| Extra 5 min to verify download | ~$1.50 |
| Premature destroy (retrain) | ~$15 + 40 min |

**ALWAYS spend the $1.50 to verify!**

---

## QUICK COMMANDS

```bash
# Check training progress
ssh -p 33762 root@ssh6.vast.ai "tail -5 /workspace/vastai_llm_training/training*.log | grep -E '(%|loss|complete)'"

# Check if GGUF exists
ssh -p 33762 root@ssh6.vast.ai "ls -la /workspace/vastai_llm_training/models/output/final/*.gguf"

# GPU utilization
ssh -p 33762 root@ssh6.vast.ai "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -3"

# Download (WAIT for completion!)
scp -P 33762 root@ssh6.vast.ai:/workspace/vastai_llm_training/models/output/final/*.gguf C:\Users\kevin\forex\forex\models\forex-r1-v3\

# Verify local file
dir "C:\Users\kevin\forex\forex\models\forex-r1-v3\*.gguf"

# ONLY THEN destroy
vastai destroy instance 30393763
```

---

## CURRENT STATUS

| Stage | Status | ETA |
|-------|--------|-----|
| SFT | ✅ COMPLETE | Done |
| DPO | 🔄 IN PROGRESS | ~3-4 min |
| GGUF Export | ⏳ PENDING | ~5 min after DPO |
| **Download** | ⏳ PENDING | ~2-3 min after export |
| **Verify** | ⏳ PENDING | After download |
| **Destroy** | ⏳ PENDING | ONLY after verify |

---

## GOLDEN RULE

```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   I WILL NOT DESTROY THE INSTANCE UNTIL I HAVE:                            ║
║                                                                            ║
║   1. ✅ Verified GGUF exists on remote (ls -la)                            ║
║   2. ✅ Downloaded GGUF to local machine (scp -P)                          ║
║   3. ✅ Verified local file exists AND size > 5GB (dir)                    ║
║   4. ✅ User confirms download complete                                    ║
║                                                                            ║
║   ONLY THEN: vastai destroy instance 30393763                              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```
