# Vast.ai Safety Rules (CRITICAL - NEVER VIOLATE)

## The Mistake That Cost Us

On 2026-01-22, training completed successfully on Vast.ai but instance was DESTROYED before download finished. Model lost. $30+ wasted. Had to retrain from scratch.

## ABSOLUTE RULES

### 1. NEVER Destroy Instance Until Download CONFIRMED

```bash
# WRONG - destroyed before download confirmed
vastai destroy instance <ID>

# RIGHT - verify file exists locally FIRST
ls -la models/forex-r1-v3/*.gguf  # MUST show file with correct size
# THEN destroy
vastai destroy instance <ID>
```

### 2. Download Verification Checklist

Before destroying ANY cloud instance:
- [ ] File downloaded to local machine
- [ ] File size matches remote (check with `ls -la`)
- [ ] File is not corrupted (can be loaded/tested)
- [ ] User CONFIRMS download complete

### 3. Use Synchronous Downloads (Not Background)

```bash
# WRONG - background download, might destroy before complete
scp ... &

# RIGHT - wait for download to finish
scp -P <port> root@<host>:/path/to/model.gguf ./models/
echo "Download complete, file size:"
ls -la ./models/*.gguf
# NOW safe to destroy
```

### 4. Training Checkpoints are NOT Recovery

Checkpoints on Vast.ai instance are LOST when instance destroyed. They don't survive. Always download final model before destroy.

### 5. Cost of Mistake

| Action | Cost |
|--------|------|
| Premature destroy | ~$30 (full retrain) |
| Extra 5 min to verify download | ~$1.50 |

**Always spend the $1.50 to verify.**

## Quick Reference

```bash
# 1. Training complete, model ready
ssh -p <port> root@<host> "ls -la /workspace/*/models/output/final/*.gguf"

# 2. Download (WAIT for completion)
scp -P <port> root@<host>:/workspace/vastai_llm_training/models/output/final/*.gguf C:\Users\kevin\forex\forex\models\forex-r1-v3\

# 3. VERIFY local file
dir "C:\Users\kevin\forex\forex\models\forex-r1-v3\"
# Must show file with size > 5GB for Q8_0

# 4. ONLY THEN destroy
vastai destroy instance <ID>
```

## Automation Script

Create download-then-destroy script:
```powershell
# download_and_destroy.ps1
param($InstanceId, $Port, $Host)

# Download
scp -P $Port "root@${Host}:/workspace/vastai_llm_training/models/output/final/*.gguf" "C:\Users\kevin\forex\forex\models\forex-r1-v3\"

# Verify
$file = Get-ChildItem "C:\Users\kevin\forex\forex\models\forex-r1-v3\*.gguf" -ErrorAction SilentlyContinue
if ($file -and $file.Length -gt 5GB) {
    Write-Host "Download verified: $($file.Name) ($($file.Length / 1GB) GB)"
    vastai destroy instance $InstanceId
    Write-Host "Instance destroyed"
} else {
    Write-Host "ERROR: Download failed or incomplete. NOT destroying instance."
}
```
