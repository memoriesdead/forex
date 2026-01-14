# Project Isolation Rules (CRITICAL)

## This Project: FOREX

```
╔═══════════════════════════════════════════════════════════╗
║  PROJECT: FOREX                                           ║
║  Local: C:\Users\kevin\forex                              ║
║  Oracle Cloud: /home/ubuntu/projects/forex/               ║
║  memory-keeper: project = "forex"                         ║
║  claude-mem: dataDir = /home/ubuntu/.claude-mem/projects/forex  ║
╚═══════════════════════════════════════════════════════════╝
```

**SSH Tunnel Required:** `localhost:37777` and `localhost:3847` → Oracle Cloud

## Other Projects on Same Infrastructure (DO NOT ACCESS)

| Project | Local Path | Oracle Path | Status |
|---------|------------|-------------|--------|
| kalshi | `C:/Users/kevin/kalshi` | `/home/ubuntu/.claude-mem/projects/kalshi` | OFF LIMITS |
| bitcoin | `C:/Users/kevin/bitcoin` | `/home/ubuntu/.claude-mem/projects/bitcoin` | OFF LIMITS |
| polymarket | `C:/Users/kevin/polymarket` | `/home/ubuntu/.claude-mem/projects/polymarket` | OFF LIMITS |
| alpaca | `C:/Users/kevin/alpaca` | `/home/ubuntu/.claude-mem/projects/alpaca` | OFF LIMITS |
| paymentcrypto | `C:/Users/kevin/paymentcrypto` | `/home/ubuntu/.claude-mem/projects/paymentcrypto` | OFF LIMITS |

## Absolute Isolation Rules

### NEVER Do These

1. **NEVER** access, read, or modify files in other project directories
2. **NEVER** query memory-keeper with project names other than `forex`
3. **NEVER** access claude-mem data in other project dataDirs
4. **NEVER** use paths containing `/kalshi/`, `/bitcoin/`, `/polymarket/`, `/alpaca/`, `/paymentcrypto/`
5. **NEVER** reference or import code from sibling project directories
6. **NEVER** store forex data in another project's storage location
7. **NEVER** create cross-project dependencies or shared utilities

### ALWAYS Do These

1. **ALWAYS** verify paths contain `/forex/` before file operations on Oracle Cloud
2. **ALWAYS** use `forex` as the project identifier in memory-keeper calls
3. **ALWAYS** store data in designated forex directories only
4. **ALWAYS** prefix logs and temp files with `forex_` to avoid collisions
5. **ALWAYS** validate Oracle Cloud paths start with `/home/ubuntu/projects/forex/`

## Path Validation Pattern

Before any Oracle Cloud file operation, validate:

```python
# Required validation before remote file operations
def validate_forex_path(path: str) -> bool:
    allowed_prefixes = [
        "/home/ubuntu/projects/forex/",
        "/home/ubuntu/.claude-mem/projects/forex/",
        "/home/ubuntu/claude-mem-storage/forex/"
    ]
    return any(path.startswith(prefix) for prefix in allowed_prefixes)
```

## Memory-Keeper Usage

```
# CORRECT - forex project only
mcp__memory-keeper__context_session_start
- projectDir: C:\Users\kevin\forex  # THIS project only

mcp__memory-keeper__context_save
- channel: forex-*  # Use forex prefix for channels

# WRONG - accessing other projects
- projectDir: C:\Users\kevin\kalshi  # NEVER
- channel: kalshi-*  # NEVER
```

## Oracle Cloud SSH Commands

```bash
# CORRECT - forex paths only
ssh ubuntu@89.168.65.47 "ls /home/ubuntu/projects/forex/"
ssh ubuntu@89.168.65.47 "cat /home/ubuntu/.claude-mem/projects/forex/data.json"

# WRONG - other project paths
ssh ubuntu@89.168.65.47 "ls /home/ubuntu/projects/kalshi/"  # NEVER
ssh ubuntu@89.168.65.47 "cat /home/ubuntu/.claude-mem/projects/bitcoin/*"  # NEVER
```

## Shared Resources (Read-Only)

The only shared resources across projects:

| Resource | Path | Access |
|----------|------|--------|
| MCP servers (code) | `/home/ubuntu/mem-server/` | Read-only, shared |
| claude-mem (code) | `/home/ubuntu/.claude-mem/claude-mem-src/` | Read-only, shared |
| System packages | `/usr/bin/`, `/home/ubuntu/.bun/` | Read-only, shared |

**Data is NEVER shared** - each project has isolated storage.

## Conflict Prevention Checklist

Before any operation, verify:

- [ ] Path contains `forex` (not another project name)
- [ ] Memory-keeper calls use `forex` project
- [ ] No imports from sibling directories
- [ ] Temp files prefixed with `forex_`
- [ ] Logs go to forex-specific directories
- [ ] No hardcoded paths to other projects

## If Conflict Detected

If you accidentally access another project:

1. **STOP immediately**
2. Do not modify any files
3. Do not save any context
4. Report the near-miss to user
5. Verify correct paths before proceeding

## Why This Matters

- 6 projects share Oracle Cloud infrastructure
- Cross-contamination corrupts project context
- Memory pollution breaks AI continuity
- Data leaks between projects cause errors
- Isolation = reliability for all projects
