# Deploy China Research Proxy to Alibaba Cloud HK VPS
# Run: powershell -File deploy_to_alibaba.ps1 -IP "YOUR_VPS_IP" -KeyPath "path/to/key.pem"

param(
    [Parameter(Mandatory=$true)]
    [string]$IP,

    [Parameter(Mandatory=$true)]
    [string]$KeyPath,

    [string]$User = "root"
)

$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "DEPLOYING CHINA RESEARCH PROXY"
Write-Host "Target: $User@$IP"
Write-Host "=========================================="

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Upload setup script
Write-Host "`n[1/3] Uploading setup script..."
scp -i $KeyPath "$ScriptDir\setup_china_proxy.sh" "${User}@${IP}:/tmp/"

# Make executable and run
Write-Host "`n[2/3] Running setup on VPS (this takes 5-10 minutes)..."
ssh -i $KeyPath "${User}@${IP}" "chmod +x /tmp/setup_china_proxy.sh && /tmp/setup_china_proxy.sh"

# Test the deployment
Write-Host "`n[3/3] Testing deployment..."
$TestResult = ssh -i $KeyPath "${User}@${IP}" "curl -s http://localhost:8888/health"
Write-Host "Health check: $TestResult"

# Update local client config
Write-Host "`n[4/4] Updating local client configuration..."
$ClientPath = "$ScriptDir\china_research_client.py"
(Get-Content $ClientPath) -replace 'YOUR_VPS_IP', $IP | Set-Content $ClientPath

Write-Host "`n=========================================="
Write-Host "DEPLOYMENT COMPLETE"
Write-Host "=========================================="
Write-Host ""
Write-Host "Your China Research Proxy is running at:"
Write-Host "  - API: http://${IP}:8888"
Write-Host "  - SOCKS5: ${IP}:1080"
Write-Host ""
Write-Host "Test commands:"
Write-Host "  curl http://${IP}:8888/health"
Write-Host "  curl 'http://${IP}:8888/search/baidu?q=量化交易'"
Write-Host ""
Write-Host "Local client usage:"
Write-Host "  python $ScriptDir\china_research_client.py health"
Write-Host "  python $ScriptDir\china_research_client.py search kalman"
Write-Host "  python $ScriptDir\china_research_client.py queries"
Write-Host ""
Write-Host "SSH tunnel for SOCKS5 proxy:"
Write-Host "  ssh -i `"$KeyPath`" -D 1080 -N ${User}@${IP}"
Write-Host ""
