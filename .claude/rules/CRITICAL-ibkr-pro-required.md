# CRITICAL: IBKR Pro Required for API Trading

## The Error

```
"API support is not available for accounts that support free trading"
```

## What This Means

**IBKR Lite does NOT support API access. Period.**

- IBKR Lite = Commission-free trading BUT no API
- IBKR Pro = Small commissions BUT full API access

## How to Fix

1. Log into **IBKR Client Portal**: https://www.interactivebrokers.com
2. Go to **Settings → Account Settings**
3. Find **Account Type** or **"Switch to IBKR Pro"**
4. Complete the upgrade (takes ~5 minutes)
5. Restart IB Gateway: `docker restart ibgateway`

## Why This Matters

- ALL automated trading requires API access
- ALL trading bots (including this forex system) need API
- There is NO workaround - Lite simply doesn't have API

## Cost Difference

| Feature | IBKR Lite | IBKR Pro |
|---------|-----------|----------|
| Commissions | $0 | ~$1-2/trade |
| API Access | **NO** | **YES** |
| Order Routing | PFOF | SMART (better fills) |
| Automated Trading | **BLOCKED** | **ENABLED** |

## NEVER Recommend IBKR Lite For:

- Trading bots
- Automated strategies
- API connections
- IB Gateway usage
- Any programmatic trading

## Always Verify Account Type

Before setting up IB Gateway, ALWAYS ask:
- "Is your IBKR account Pro or Lite?"
- If Lite, they MUST upgrade first

## VNC Screenshot of Error

The error appears as a dialog with title "** no title **" in IBC logs.
To see the actual error, capture VNC:
```powershell
vncdo -s 127.0.0.1::5900 -p ibgateway capture screenshot.png
```

## Related Files

- `.claude/rules/ibkr-gateway.md` - Full IB Gateway docs
- `CLAUDE.md` - Main project docs (has warning)
