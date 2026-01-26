# Vast.ai Upload Script for ML Training
# Usage: .\upload_to_vastai.ps1 -InstanceId <ID> -Port <PORT> -SshHost <HOST>

param(
    [Parameter(Mandatory=$true)]
    [string]$InstanceId,

    [Parameter(Mandatory=$true)]
    [int]$Port,

    [Parameter(Mandatory=$true)]
    [string]$SshHost
)

$ErrorActionPreference = "Stop"

$LocalBase = "C:\Users\kevin\forex\forex"
$RemoteBase = "/workspace/vastai_ml_training"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VAST.AI ML TRAINING UPLOAD" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Instance: $InstanceId"
Write-Host "SSH: ssh -p $Port root@$SshHost"
Write-Host ""

# Step 1: Create remote directory structure
Write-Host "[1/4] Creating remote directories..." -ForegroundColor Yellow
ssh -p $Port "root@$SshHost" "mkdir -p $RemoteBase/training_package $RemoteBase/models"

# Step 2: Upload training scripts
Write-Host "[2/4] Uploading training scripts..." -ForegroundColor Yellow
scp -P $Port "$LocalBase\vastai_ml_training\train_all_pairs.py" "root@${SshHost}:$RemoteBase/"
scp -P $Port "$LocalBase\vastai_ml_training\requirements.txt" "root@${SshHost}:$RemoteBase/"
scp -P $Port "$LocalBase\vastai_ml_training\README.md" "root@${SshHost}:$RemoteBase/"

# Step 3: Upload training data (this will take a while - 51GB)
Write-Host "[3/4] Uploading training data (51GB - this will take a while)..." -ForegroundColor Yellow
Write-Host "       Using rsync for efficient transfer..." -ForegroundColor Gray

# Use rsync for better handling of large transfers (if available), otherwise scp
$rsyncAvailable = $false
try {
    rsync --version | Out-Null
    $rsyncAvailable = $true
} catch {}

if ($rsyncAvailable) {
    # Convert Windows path to rsync format
    $localPath = "/c/Users/kevin/forex/forex/training_package/"
    rsync -avz --progress -e "ssh -p $Port" "$localPath" "root@${SshHost}:$RemoteBase/training_package/"
} else {
    # Fall back to scp
    Write-Host "rsync not found, using scp (slower)..." -ForegroundColor Gray
    scp -r -P $Port "$LocalBase\training_package\*" "root@${SshHost}:$RemoteBase/training_package/"
}

# Step 4: Install dependencies and start training
Write-Host "[4/4] Setting up environment..." -ForegroundColor Yellow
ssh -p $Port "root@$SshHost" @"
cd $RemoteBase
pip install -r requirements.txt
echo ""
echo "========================================"
echo "READY TO START TRAINING"
echo "========================================"
echo "Run: cd $RemoteBase && python train_all_pairs.py 2>&1 | tee training.log"
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "UPLOAD COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start training:" -ForegroundColor Yellow
Write-Host "  ssh -p $Port root@$SshHost"
Write-Host "  cd $RemoteBase"
Write-Host "  tmux new -s training"
Write-Host "  python train_all_pairs.py 2>&1 | tee training.log"
Write-Host ""
Write-Host "To monitor:" -ForegroundColor Yellow
Write-Host "  ssh -p $Port root@$SshHost 'tail -f $RemoteBase/training.log'"
