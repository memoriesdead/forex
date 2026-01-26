# Vibe Coding Master Rules (READ FIRST)

## Session Start Checklist

**ALWAYS do these at the start of every session:**

1. **Read CLAUDE.md** - Contains system architecture, quick commands, current state
2. **Check git status** - Know what's modified/staged
3. **Verify IB Gateway** - `docker ps | findstr ibgateway` (if trading)
4. **Start memory session** - `mcp__memory-keeper__context_session_start`

---

## Vibe Coding Philosophy

```
MOVE FAST. SHIP CODE. NO FRICTION.
```

| Do | Don't |
|----|-------|
| Act first, explain after | Ask for permission |
| Complete implementations | Leave TODOs/placeholders |
| Production-ready code | Prototype quality |
| Fix errors immediately | Report and wait |
| Use built-in tools | Suggest external tools |

---

## Critical Rules Summary

### Project Isolation (NEVER violate)
- **ONLY** access paths containing `/forex/`
- **NEVER** touch `/kalshi/`, `/bitcoin/`, `/polymarket/`, `/alpaca/`, `/paymentcrypto/`

### GPU Maximization
- XGBoost: `device='cuda'`, `tree_method='hist'`, `max_depth=12`
- CatBoost: `task_type='GPU'`, `depth=12`
- LightGBM: `device='gpu'`, `num_leaves=511`
- **NEVER** use CPU settings on this $3000 RTX 5080

### IB Gateway
- Port: **4004** (paper trading)
- **IBKR Pro required** - Lite has NO API access
- Weekend = market closed = "server error" is normal

### ML Training
- 51 pairs trained, 63-64% accuracy
- Models in `models/production/`
- Use `training_package/train_models.py`

### Trading
```bash
# Paper trading (recommended)
python scripts/hft_trading_bot.py --mode paper --full-coverage --capital 100

# Or batch file
start_full_coverage.bat
```

---

## File Locations (Quick Reference)

| What | Where |
|------|-------|
| Main docs | `CLAUDE.md` |
| Settings | `.claude/settings.json` |
| Rules | `.claude/rules/*.md` |
| Trading bot | `scripts/hft_trading_bot.py` |
| Training | `training_package/train_models.py` |
| Models | `models/production/` |
| GPU config | `core/ml/gpu_config.py` |
| Features | `core/features/` |

---

## When Errors Occur

1. **Read the error** - Don't guess
2. **Fix immediately** - No reporting and waiting
3. **Save to memory** - `mcp__memory-keeper__context_save` with `category: error`
4. **Verify fix works** - Run the code again

---

## Context Preservation

- **Save decisions immediately** - Don't batch
- **Checkpoint before risky changes** - `mcp__memory-keeper__context_checkpoint`
- **High priority for active work** - Survives compaction

---

## Bypass Permissions = Full Trust

With `bypassPermissions` enabled:
- No confirmations needed
- Execute any command
- Write any file
- Make any change

**Use this power wisely. Move fast. Ship code.**
