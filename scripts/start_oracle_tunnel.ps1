# Oracle Cloud SSH Tunnel - Background Starter
# Starts tunnel hidden, auto-restarts on failure
# Forwards: MCP servers ONLY (IB Gateway is LOCAL Docker)

$keyPath = "C:\Users\kevin\forex\forex\ssh-key-89.key"
$host_ip = "89.168.65.47"

# Check if tunnel already running
$existing = Get-Process ssh -ErrorAction SilentlyContinue | Where-Object {
    (Get-WmiObject Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -like "*37777*"
}

if ($existing) {
    Write-Host "Tunnel already running (PID: $($existing.Id))"
    exit 0
}

Write-Host "Starting Oracle Cloud SSH tunnel..."
Write-Host "Forwarding:"
Write-Host "  - 37777 -> claude-mem (MCP)"
Write-Host "  - 3847  -> memory-keeper (MCP)"
Write-Host "  - 8081  -> execution-optimizer (MCP)"
Write-Host "  - 8082  -> rl-server (MCP) - US Quant Gold Standard RL"
Write-Host ""
Write-Host "NOTE: IB Gateway runs LOCALLY on Docker (port 4004)"

# Start tunnel hidden with MCP ports only
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "ssh"
$psi.Arguments = "-L 37777:localhost:37777 -L 3847:localhost:3847 -L 8081:localhost:8081 -L 8082:localhost:8082 -i `"$keyPath`" -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -N ubuntu@$host_ip"
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$psi.CreateNoWindow = $true

$process = [System.Diagnostics.Process]::Start($psi)
Write-Host "Tunnel started (PID: $($process.Id))"

# Test connection
Start-Sleep -Seconds 2
try {
    $response = Invoke-RestMethod -Uri "http://localhost:37777/health" -TimeoutSec 5
    Write-Host "Claude-mem connected: $($response.status)"
} catch {
    Write-Host "Warning: Could not verify claude-mem connection"
}

Write-Host ""
Write-Host "IB Gateway: LOCAL Docker on port 4004 (docker start ibgateway)"
