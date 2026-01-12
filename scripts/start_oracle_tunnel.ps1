# Oracle Cloud SSH Tunnel - Background Starter
# Starts tunnel hidden, auto-restarts on failure
# Forwards: MCP servers + IB Gateway

$keyPath = "C:\Users\kevin\forex\ssh-key-2026-01-07 (1).key"
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
Write-Host "  - 37777 -> claude-mem"
Write-Host "  - 3847  -> memory-keeper"
Write-Host "  - 4001  -> IB Gateway API"
Write-Host "  - 5900  -> IB Gateway VNC"

# Start tunnel hidden with all ports
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "ssh"
$psi.Arguments = "-L 37777:localhost:37777 -L 3847:localhost:3847 -L 4001:localhost:4001 -L 5900:localhost:5900 -i `"$keyPath`" -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -N ubuntu@$host_ip"
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
Write-Host "IB Gateway: Connect trading bot to localhost:4001"
Write-Host "VNC Access: Connect VNC client to localhost:5900 (password: ibgateway)"
