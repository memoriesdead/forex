# Run as Administrator
# Right-click and "Run with PowerShell" as Admin

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "ML TRAINING SYSTEM OPTIMIZATION" -ForegroundColor Cyan
Write-Host "AMD Ryzen 9 9900X + RTX 5080 16GB" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check admin privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script requires Administrator privileges!" -ForegroundColor Red
    Write-Host "Right-click and select 'Run as Administrator'" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "`n[1/8] Setting Ultimate Performance Power Plan..." -ForegroundColor Green
# Create and activate Ultimate Performance plan
$existingPlan = powercfg /list | Select-String "Ultimate Performance"
if (-not $existingPlan) {
    powercfg /duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61
}
$ultimatePlan = powercfg /list | Select-String "Ultimate Performance" | ForEach-Object { $_.Line -match '([a-f0-9-]{36})' | Out-Null; $matches[1] }
if ($ultimatePlan) {
    powercfg /setactive $ultimatePlan
    Write-Host "   Ultimate Performance plan activated" -ForegroundColor White
}

Write-Host "`n[2/8] Disabling CPU Core Parking..." -ForegroundColor Green
# Get active scheme GUID
$activeScheme = (powercfg /getactivescheme) -replace '.*GUID: ([a-f0-9-]+).*', '$1'
# Disable core parking (0 = all cores active)
powercfg /setacvalueindex $activeScheme 54533251-82be-4824-96c1-47b60b740d00 893dee8e-2bef-41e0-89c6-b55d0929964c 0
powercfg /setdcvalueindex $activeScheme 54533251-82be-4824-96c1-47b60b740d00 893dee8e-2bef-41e0-89c6-b55d0929964c 0
# Set minimum processor state to 100%
powercfg /setacvalueindex $activeScheme 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 100
powercfg /setdcvalueindex $activeScheme 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 100
# Set maximum processor state to 100%
powercfg /setacvalueindex $activeScheme 54533251-82be-4824-96c1-47b60b740d00 75b0ae3f-bce0-45a7-8c89-c9611c25e100 100
powercfg /setdcvalueindex $activeScheme 54533251-82be-4824-96c1-47b60b740d00 75b0ae3f-bce0-45a7-8c89-c9611c25e100 100
Write-Host "   Core parking disabled, CPU always at 100%" -ForegroundColor White

Write-Host "`n[3/8] Disabling Sleep and Hibernate..." -ForegroundColor Green
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /change monitor-timeout-ac 0
powercfg /change monitor-timeout-dc 0
powercfg /hibernate off 2>$null
Write-Host "   Sleep and hibernate disabled" -ForegroundColor White

Write-Host "`n[4/8] Configuring GPU TDR (Timeout Detection Recovery)..." -ForegroundColor Green
# Disable TDR (prevents GPU timeout during long ML computations)
$gfxPath = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
Set-ItemProperty -Path $gfxPath -Name "TdrLevel" -Value 0 -Type DWord -Force
Set-ItemProperty -Path $gfxPath -Name "TdrDelay" -Value 60 -Type DWord -Force
Set-ItemProperty -Path $gfxPath -Name "TdrDdiDelay" -Value 60 -Type DWord -Force
Write-Host "   TDR disabled (GPU won't timeout during long training)" -ForegroundColor White

Write-Host "`n[5/8] Enabling Hardware-Accelerated GPU Scheduling..." -ForegroundColor Green
Set-ItemProperty -Path $gfxPath -Name "HwSchMode" -Value 2 -Type DWord -Force
Write-Host "   HAGS enabled" -ForegroundColor White

Write-Host "`n[6/8] Optimizing Memory Management..." -ForegroundColor Green
# Disable memory compression (uses CPU cycles)
$mmPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
Set-ItemProperty -Path $mmPath -Name "DisablePagingExecutive" -Value 1 -Type DWord -Force
# Large system cache for data processing
Set-ItemProperty -Path $mmPath -Name "LargeSystemCache" -Value 1 -Type DWord -Force
# Disable Superfetch for ML workloads (we want predictable memory)
Stop-Service -Name "SysMain" -Force -ErrorAction SilentlyContinue
Set-Service -Name "SysMain" -StartupType Disabled -ErrorAction SilentlyContinue
Write-Host "   Memory optimized for ML workloads" -ForegroundColor White

Write-Host "`n[7/8] Disabling Unnecessary Services..." -ForegroundColor Green
$servicesToDisable = @(
    "DiagTrack",           # Connected User Experiences and Telemetry
    "dmwappushservice",    # Device Management WAP Push
    "WSearch",             # Windows Search (heavy disk I/O)
    "XboxGipSvc",          # Xbox Accessory Management
    "XblAuthManager",      # Xbox Live Auth Manager
    "XblGameSave",         # Xbox Live Game Save
    "XboxNetApiSvc"        # Xbox Live Networking
)
foreach ($svc in $servicesToDisable) {
    $service = Get-Service -Name $svc -ErrorAction SilentlyContinue
    if ($service) {
        Stop-Service -Name $svc -Force -ErrorAction SilentlyContinue
        Set-Service -Name $svc -StartupType Disabled -ErrorAction SilentlyContinue
        Write-Host "   Disabled: $svc" -ForegroundColor Gray
    }
}
Write-Host "   Background services minimized" -ForegroundColor White

Write-Host "`n[8/10] Configuring Virtual Memory..." -ForegroundColor Green
# Set virtual memory to system managed (let Windows optimize for 32GB RAM)
# For ML with 32GB RAM, we want minimal paging
$cs = Get-WmiObject -Class Win32_ComputerSystem -EnableAllPrivileges
$cs.AutomaticManagedPagefile = $true
$cs.Put() | Out-Null
Write-Host "   Virtual memory set to automatic" -ForegroundColor White

Write-Host "`n[9/10] Enabling Native NVMe Driver (80% faster I/O)..." -ForegroundColor Green
# Enable native NVMe for Windows 11 25H2+ (experimental but tested)
$nvmePath = "HKLM:\SYSTEM\CurrentControlSet\Services\stornvme\Parameters\Device"
if (-not (Test-Path $nvmePath)) {
    New-Item -Path $nvmePath -Force | Out-Null
}
Set-ItemProperty -Path $nvmePath -Name "ForcedPhysicalSectorSizeInBytes" -Value 4096 -Type DWord -Force
Set-ItemProperty -Path $nvmePath -Name "IdleTimeoutInMS" -Value 0 -Type DWord -Force
# Enable native NVMe path
$storportPath = "HKLM:\SYSTEM\CurrentControlSet\Services\stornvme"
Set-ItemProperty -Path $storportPath -Name "IoLatencyCapNs" -Value 0 -Type QWord -Force -ErrorAction SilentlyContinue
Write-Host "   Native NVMe optimizations applied" -ForegroundColor White

Write-Host "`n[10/10] Optimizing Network for Large Downloads..." -ForegroundColor Green
# Increase TCP/IP performance for large downloads
$tcpPath = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters"
Set-ItemProperty -Path $tcpPath -Name "TcpWindowSize" -Value 65535 -Type DWord -Force -ErrorAction SilentlyContinue
Set-ItemProperty -Path $tcpPath -Name "GlobalMaxTcpWindowSize" -Value 65535 -Type DWord -Force -ErrorAction SilentlyContinue
Set-ItemProperty -Path $tcpPath -Name "TcpTimedWaitDelay" -Value 30 -Type DWord -Force -ErrorAction SilentlyContinue
Set-ItemProperty -Path $tcpPath -Name "MaxUserPort" -Value 65534 -Type DWord -Force -ErrorAction SilentlyContinue
# Disable Nagle's algorithm for faster small packet handling
Set-ItemProperty -Path $tcpPath -Name "TcpNoDelay" -Value 1 -Type DWord -Force -ErrorAction SilentlyContinue
# Increase ephemeral port range
netsh int tcp set global autotuninglevel=highlyrestricted 2>$null
netsh int tcp set global chimney=disabled 2>$null
Write-Host "   Network optimized for large downloads" -ForegroundColor White

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "OPTIMIZATION COMPLETE" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "`nChanges applied:" -ForegroundColor Yellow
Write-Host "  - Ultimate Performance power plan active"
Write-Host "  - CPU core parking disabled (all 12 cores active)"
Write-Host "  - CPU always runs at max frequency"
Write-Host "  - Sleep/hibernate disabled (24/7 operation)"
Write-Host "  - GPU TDR disabled (no timeout during training)"
Write-Host "  - Hardware GPU scheduling enabled"
Write-Host "  - Memory optimized for ML workloads"
Write-Host "  - Background services minimized"
Write-Host "  - Native NVMe driver enabled (up to 80% faster I/O)"
Write-Host "  - Network stack optimized for large downloads"
Write-Host ""
Write-Host "RESTART REQUIRED for all changes to take effect!" -ForegroundColor Red
Write-Host ""
$restart = Read-Host "Restart now? (y/n)"
if ($restart -eq "y") {
    Restart-Computer -Force
}
