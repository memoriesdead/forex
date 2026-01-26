# Download trained models from Vast.ai
# Usage: .\download_models.ps1 -Port <PORT> -SshHost <HOST>

param(
    [Parameter(Mandatory=$true)]
    [int]$Port,

    [Parameter(Mandatory=$true)]
    [string]$SshHost
)

$ErrorActionPreference = "Stop"

$LocalModels = "C:\Users\kevin\forex\forex\models\production"
$RemoteModels = "/workspace/vastai_ml_training/models"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DOWNLOADING TRAINED MODELS FROM VAST.AI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check what's available
Write-Host "[1/3] Checking trained models..." -ForegroundColor Yellow
$remoteCount = ssh -p $Port "root@$SshHost" "ls $RemoteModels/*.pkl 2>/dev/null | wc -l"
Write-Host "       Found $remoteCount model files" -ForegroundColor Gray

# Check feature counts
Write-Host "[2/3] Verifying feature counts..." -ForegroundColor Yellow
ssh -p $Port "root@$SshHost" @"
cd $RemoteModels
for f in *_results.json; do
    pair=\$(basename "\$f" _results.json)
    feat=\$(python3 -c "import json; d=json.load(open('\$f')); print(d.get('_meta',{}).get('feature_count',0))")
    acc=\$(python3 -c "import json; d=json.load(open('\$f')); print(f\"{d.get('target_direction_10',{}).get('accuracy',0):.4f}\")")
    echo "  \$pair: \$feat features, ACC=\$acc"
done
"@

# Download
Write-Host "[3/3] Downloading models..." -ForegroundColor Yellow
scp -P $Port "root@${SshHost}:$RemoteModels/*.pkl" "$LocalModels\"
scp -P $Port "root@${SshHost}:$RemoteModels/*.json" "$LocalModels\"

# Verify local
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "DOWNLOAD COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

$localCount = (Get-ChildItem "$LocalModels\*_models.pkl").Count
Write-Host "Downloaded $localCount model files to:" -ForegroundColor Gray
Write-Host "  $LocalModels" -ForegroundColor Gray

# Count 1239 feature models
$with1239 = 0
Get-ChildItem "$LocalModels\*_results.json" | ForEach-Object {
    $json = Get-Content $_.FullName | ConvertFrom-Json
    if ($json._meta.feature_count -eq 1239) { $with1239++ }
}
Write-Host ""
Write-Host "Models with 1239 features: $with1239" -ForegroundColor $(if ($with1239 -ge 50) { "Green" } else { "Yellow" })
