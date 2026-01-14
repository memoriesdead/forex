# Remote Infrastructure Architecture

## Overview

**Project: FOREX** (C:\Users\kevin\forex)

This project runs on a **distributed architecture** via SSH tunnel to Oracle Cloud:

```
Local (Your PC)                    Oracle Cloud (89.168.65.47)
    │                                      │
    ├─ localhost:37777 ───SSH tunnel────► claude-mem (auto-capture)
    ├─ localhost:3847  ───SSH tunnel────► memory-keeper (manual saves)
    ├─ localhost:4001  ───SSH tunnel────► IB Gateway API (trading)
    ├─ localhost:5900  ───SSH tunnel────► IB Gateway VNC (manual login)
    │                                      │
    └─ Claude Code ◄────── MCP ─────────► Both servers
```

**Required before each session:**
```powershell
powershell -File C:\Users\kevin\forex\scripts\start_oracle_tunnel.ps1
```

## IB Gateway (Trading)

| Setting | Value |
|---------|-------|
| Container | ibgateway |
| Account | DUO423364 (paper) |
| API Port | 4001 |
| VNC Port | 5900 |
| VNC Password | ibgateway |
| Location | /home/ubuntu/ibgateway/ |

**Connect trading bot locally:**
```powershell
ssh -i "C:\Users\kevin\forex\ssh-key-2026-01-07 (1).key" -L 4001:localhost:4001 ubuntu@89.168.65.47 -N
```

**Check status:**
```bash
ssh ubuntu@89.168.65.47 "docker logs ibgateway | tail -10"
```

## Local Configuration Files

| File | Purpose |
|------|---------|
| `~/.claude-mem/settings.json` | Worker connection (port 37777) |
| `~/.claude/hooks.json` | Auto-capture hooks for claude-mem |
| `~/.claude-mem/scripts/*` | Hook scripts (copied from Oracle Cloud) |
| `scripts/start_oracle_tunnel.ps1` | Background tunnel starter |
| `scripts/oracle_tunnel.bat` | Manual tunnel (keeps window open) |

## CRITICAL: Multi-Project Isolation

**Oracle Cloud hosts 6 INDEPENDENT projects. This is FOREX only.**

```
Oracle Cloud (89.168.65.47)
├── /home/ubuntu/projects/forex/        ← THIS PROJECT ONLY
├── /home/ubuntu/.claude-mem/projects/
│   ├── forex/                          ← THIS PROJECT ONLY
│   ├── kalshi/                         ← OFF LIMITS
│   ├── bitcoin/                        ← OFF LIMITS
│   ├── polymarket/                     ← OFF LIMITS
│   ├── alpaca/                         ← OFF LIMITS
│   └── paymentcrypto/                  ← OFF LIMITS
└── /home/ubuntu/claude-mem-storage/
    ├── forex/                          ← THIS PROJECT ONLY
    ├── kalshi/                         ← OFF LIMITS
    └── [other projects]/               ← OFF LIMITS
```

### Isolation Rules

1. **ONLY** access paths containing `/forex/`
2. **NEVER** read/write to other project directories
3. **NEVER** query memory for other projects
4. **NEVER** use other project names in MCP calls
5. **ALWAYS** validate paths before Oracle Cloud operations

### Shared Services (Code Only - Data Isolated)

| Service | Port | Shared? | Data Isolated? |
|---------|------|---------|----------------|
| memory-keeper | 3847 | Yes (code) | Yes (per-project dirs) |
| claude-mem worker | 37777 | Yes (code) | Yes (per-project dirs) |
| SSH | 22 | Yes | N/A |

**Services are shared, DATA is isolated.** Each project has its own storage directory.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    LOCAL MACHINE                        │
│                                                         │
│  • Claude Code CLI (thin client)                        │
│  • SSH connections                                      │
│  • Minimal resource usage                               │
│                                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ SSH tunnel (89.168.65.47)
                   │
┌──────────────────▼──────────────────────────────────────┐
│              ORACLE CLOUD INSTANCE                      │
│          /home/ubuntu/projects/forex/                   │
│                                                         │
│  ┌─────────────────────────────────────────┐            │
│  │ MCP Servers                             │            │
│  │ • memory-keeper (context management)    │            │
│  │ • claude-mem (automatic capture)        │            │
│  │ • filesystem (file access)              │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
│  ┌─────────────────────────────────────────┐            │
│  │ Data Storage                            │            │
│  │ • data/ (raw forex data)                │            │
│  │ • data_cleaned/ (processed)             │            │
│  │ • models/ (trained models)              │            │
│  │ • logs/ (training logs)                 │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
│  ┌─────────────────────────────────────────┐            │
│  │ Compute                                 │            │
│  │ • Model training (remote_train.py)      │            │
│  │ • Data processing                       │            │
│  │ • MCP server execution                  │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ GitHub API (HTTPS)
                   │
┌──────────────────▼──────────────────────────────────────┐
│                 GITHUB REPOSITORY                       │
│              (forex-memory - private)                   │
│                                                         │
│  • Memory/context storage (JSON files)                  │
│  • Git history (automatic backups)                      │
│  • Searchable via GitHub API                            │
│  • Zero local storage                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Resource Distribution

### Local Machine (Minimal)
**Runs:**
- Claude Code CLI only

**Resources:**
- CPU: ~5% (SSH + Claude CLI)
- RAM: ~100MB
- Disk: ~50MB (code only, no data/models/memory)

**Does NOT run:**
- MCP servers
- Memory databases
- Data storage
- Model training

### Oracle Cloud (Primary Compute)
**Specs:**
- Host: 89.168.65.47
- CPU: ARM64 (aarch64)
- RAM: 12GB
- Disk: 45GB (42GB available)
- OS: Ubuntu 24.04

**Runs:**
- All MCP servers (memory-keeper, claude-mem, etc.)
- Data storage (data/, data_cleaned/, models/, logs/)
- Model training (remote_train.py)
- Heavy compute tasks

**Location:** `/home/ubuntu/projects/forex/`

**Isolation:** Shared with 5 other projects, path validation enforced

### GitHub (Memory Persistence)
**Repository:** Private `forex-memory` repo

**Stores:**
- All memory/context data (JSON files)
- Git history (backup)
- Searchable index

**Access:** GitHub API via token (GITHUB_TOKEN in .env)

**Structure:**
```
forex-memory/
├── memories/
│   ├── 2026-01/
│   │   ├── decisions/
│   │   ├── progress/
│   │   ├── notes/
│   │   └── errors/
│   └── index.json
└── README.md
```

## Data Flow Examples

### Example 1: Save Context to Memory

```
1. You: Use mcp__memory-keeper__context_save
2. Claude CLI (local) → SSH tunnel → MCP server (Oracle Cloud)
3. MCP server creates JSON file
4. MCP server → GitHub API → Commit to forex-memory repo
5. GitHub stores permanently
6. Response ← SSH tunnel ← You
```

**Local resources used:** Minimal (just SSH tunnel)
**Heavy lifting:** Oracle Cloud + GitHub

### Example 2: Search Context

```
1. You: Use mcp__memory-keeper__context_search
2. Claude CLI (local) → SSH tunnel → MCP server (Oracle Cloud)
3. MCP server → GitHub API → Fetch matching files
4. MCP server processes/filters results
5. Results ← SSH tunnel ← You
```

**Local resources used:** Minimal (just SSH tunnel)
**Heavy lifting:** Oracle Cloud + GitHub

### Example 3: Train Model

```
1. You: python3 scripts/remote_train.py run train.py
2. SSH to Oracle Cloud
3. Upload train.py to /home/ubuntu/projects/forex/
4. Execute training on Oracle Cloud (uses local data/ already there)
5. Model saved to /home/ubuntu/projects/forex/models/
6. Optionally pull model to local machine
```

**Local resources used:** Zero (except SSH connection)
**Heavy lifting:** Oracle Cloud (full training compute)

### Example 4: Sync Data

```
1. You: python3 scripts/oracle_sync.py push data
2. rsync over SSH → Oracle Cloud
3. Data stored in /home/ubuntu/projects/forex/data/
4. Local data/ can be deleted to free space
```

**Local resources used:** Minimal (rsync client)
**Heavy lifting:** Oracle Cloud (stores all data)

## Why This Architecture?

**Problem:**
- Local machine couldn't handle:
  - MCP servers (CPU/RAM intensive)
  - Memory storage (disk space)
  - Large datasets (disk space)
  - Model training (compute intensive)

**Solution:**
- Offload everything to Oracle Cloud + GitHub
- Local machine is just a thin client
- All heavy operations run remotely

**Benefits:**
1. **Local Resources:** 95%+ reduction in usage
2. **Scalability:** Oracle Cloud can handle more than local
3. **Persistence:** GitHub provides redundant backup
4. **Accessibility:** Access from any machine
5. **Cost:** $0 (free tier + free GitHub)

## Scripts for Remote Operations

**Data sync:**
- `scripts/oracle_sync.py` - Bidirectional data sync
- `scripts/oracle.sh` - Quick commands wrapper

**Remote training:**
- `scripts/remote_train.py` - Execute training on Oracle Cloud

**MCP management:**
- `scripts/setup_remote_mcp.py` - Initial MCP server setup
- `mcp_servers/mcp_manager.sh` - Start/stop MCP servers (on Oracle Cloud)

**Python integration:**
- `scripts/oracle_helper.py` - Oracle Cloud operations in Python

## Connection Details

**Oracle Cloud SSH:**
```bash
Host: 89.168.65.47
User: ubuntu
Key: ssh-key-2026-01-07 (1).key
Path: /home/ubuntu/projects/forex/
```

**GitHub Memory:**
```bash
Repo: forex-memory (private)
Token: GITHUB_TOKEN (in .env)
API: https://api.github.com
```

**MCP Servers:**
```bash
Location: /home/ubuntu/projects/forex/mcp_servers/
Manager: mcp_manager.sh
Logs: /home/ubuntu/projects/forex/logs/mcp/
```

## Transparent to User

**From your perspective:**
- Claude Code works exactly the same
- MCP servers respond the same
- Memory operations work the same
- No extra steps needed

**Behind the scenes:**
- Everything runs on Oracle Cloud
- Memory stored in GitHub
- Local machine just proxies requests

**Zero configuration after setup:**
- MCP servers auto-start on Oracle Cloud
- SSH tunnel established automatically
- GitHub API calls handled by MCP servers
- You just use Claude Code normally

## Failure Modes & Recovery

**If Oracle Cloud unreachable:**
- Claude Code will timeout trying to connect to MCP servers
- Local work continues (code editing)
- Memory operations queue or fail gracefully

**If GitHub API down:**
- MCP servers can't save/retrieve memory
- Temporary cache on Oracle Cloud (if implemented)
- Resume when GitHub API returns

**If SSH connection drops:**
- MCP server connections lost
- Re-establish connection
- MCP servers still running on Oracle Cloud

## Monitoring & Management

**Check MCP server status:**
```bash
bash scripts/oracle.sh ssh "/home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh status"
```

**View MCP logs:**
```bash
bash scripts/oracle.sh ssh "/home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh logs-memory"
```

**Check Oracle Cloud resources:**
```bash
python3 scripts/oracle_sync.py status
```

**View GitHub memory:**
```
Visit: https://github.com/USERNAME/forex-memory
```

## Security

**SSH Key:**
- Local: `ssh-key-2026-01-07 (1).key` (600 permissions)
- Never committed to git (.gitignore)

**GitHub Token:**
- Stored in .env (never committed)
- Full repo access to forex-memory
- Rotate periodically

**Oracle Cloud Access:**
- SSH key authentication only (no passwords)
- Firewall rules (if configured)

**GitHub Repository:**
- MUST be private (contains project context)
- Access via PAT only

## Cost Analysis

**Oracle Cloud:**
- Free tier (already allocated)
- No additional cost

**GitHub:**
- Private repo: Free
- API calls: Free (within limits)
- Storage: Free (unlimited for standard repos)

**Total:** $0/month

## Next Architecture Improvements

**Potential future enhancements:**
1. Auto-restart MCP servers on Oracle Cloud reboot
2. MCP server health checks and alerts
3. Backup Oracle Cloud data to additional locations
4. Load balancing if running multiple projects
5. CI/CD integration for automated deployments
