# Safe Vast.ai Model Download Script
# ALWAYS use this instead of manual scp + destroy

param(
    [Parameter(Mandatory=$true)]
    [string]$InstanceId,

    [Parameter(Mandatory=$true)]
    [int]$Port,

    [Parameter(Mandatory=$true)]
    [string]$SshHost,

    [string]$RemotePath = "/workspace/latitude_h100_training/models/output/final/*.gguf",
    [string]$LocalPath = "C:\Users\kevin\forex\forex\models\forex-r1-v3"
)

Write-Host "╔════════════════════════════════════════════════════════════════════════╗"
Write-Host "║  SAFE VAST.AI MODEL DOWNLOAD                                           ║"
Write-Host "╚════════════════════════════════════════════════════════════════════════╝"

# Create local directory if needed
if (!(Test-Path $LocalPath)) {
    New-Item -ItemType Directory -Path $LocalPath -Force | Out-Null
    Write-Host "Created directory: $LocalPath"
}

# Check remote file exists and get size
Write-Host "`nChecking remote file..."
$remoteCheck = ssh -o StrictHostKeyChecking=no -p $Port "root@$SshHost" "ls -la $RemotePath 2>/dev/null"
if (!$remoteCheck) {
    Write-Host "ERROR: Remote file not found at $RemotePath"
    Write-Host "NOT destroying instance."
    exit 1
}
Write-Host "Remote file found:"
Write-Host $remoteCheck

# Download
Write-Host "`nDownloading model (this will take a few minutes)..."
$startTime = Get-Date
scp -o StrictHostKeyChecking=no -P $Port "root@${SshHost}:$RemotePath" $LocalPath
$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

# Verify local file
Write-Host "`nVerifying download..."
$localFiles = Get-ChildItem "$LocalPath\*.gguf" -ErrorAction SilentlyContinue

if (!$localFiles) {
    Write-Host "ERROR: No .gguf files found in $LocalPath"
    Write-Host "Download may have failed. NOT destroying instance."
    exit 1
}

foreach ($file in $localFiles) {
    $sizeGB = [math]::Round($file.Length / 1GB, 2)
    Write-Host "  $($file.Name): $sizeGB GB"

    if ($file.Length -lt 1GB) {
        Write-Host "WARNING: File seems too small (< 1GB). May be incomplete."
        Write-Host "NOT destroying instance. Please verify manually."
        exit 1
    }
}

Write-Host "`n╔════════════════════════════════════════════════════════════════════════╗"
Write-Host "║  DOWNLOAD VERIFIED SUCCESSFULLY                                        ║"
Write-Host "╚════════════════════════════════════════════════════════════════════════╝"
Write-Host "Duration: $([math]::Round($duration, 1)) seconds"

# Confirm before destroy
Write-Host "`nReady to destroy instance $InstanceId to stop billing."
$confirm = Read-Host "Type 'DESTROY' to confirm"

if ($confirm -eq "DESTROY") {
    Write-Host "Destroying instance $InstanceId..."
    vastai destroy instance $InstanceId
    Write-Host "Instance destroyed. Billing stopped."
} else {
    Write-Host "Destroy cancelled. Instance still running (billing continues)."
    Write-Host "To destroy manually: vastai destroy instance $InstanceId"
}
