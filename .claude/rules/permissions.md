# Permission Controls

## Quick Toggle (During Session)

**Keyboard shortcut:**
- `Ctrl+P` - Toggle bypass permissions on/off
- Works instantly during any session

## Command Line Start

**Bypass mode (default):**
```bash
claude
# or explicitly
claude --bypass-permissions
```

**Ask mode (need approval for actions):**
```bash
claude --ask-permissions
```

## Current Settings

**Default mode:** `bypassPermissions`
- No confirmations for file operations
- No confirmations for commands
- No confirmations for network requests
- Full automation enabled

## When to Toggle

**Use bypass mode (Ctrl+P ON):**
- Vibe coding - move fast
- Building features
- Rapid iteration
- Trust Claude fully

**Use ask mode (Ctrl+P OFF):**
- Reviewing unfamiliar code
- Working with production systems
- Want to review each action
- Learning what Claude is doing

## Toggle Script

Created `scripts/toggle_permissions.py` for command-line toggling outside of sessions.

**Usage:**
```bash
# Switch to bypass mode
python scripts/toggle_permissions.py bypass

# Switch to ask mode
python scripts/toggle_permissions.py ask

# Check current mode
python scripts/toggle_permissions.py status
```

## Settings Explained

**bypass mode** = Claude does everything automatically
**ask mode** = Claude asks before each action

**In session:** Press `Ctrl+P` to toggle instantly
**Before session:** Use command line flags or toggle script
