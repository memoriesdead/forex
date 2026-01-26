# IBKR Gateway (SHARED Docker - Stocks + Forex)

## Multi-Market Architecture (2026-01-23)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              SHARED IB GATEWAY - STOCKS & FOREX SIMULTANEOUS                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ONE container, DIFFERENT clientIds = both can trade at same time:           ║
║                                                                              ║
║  ┌─────────────────┐                              ┌─────────────────┐       ║
║  │  FOREX          │     ┌──────────────────┐     │   STOCKS        │       ║
║  │  clientId=100   │────▶│   IB GATEWAY     │◀────│   clientId=5    │       ║
║  │                 │     │   Docker:4004    │     │                 │       ║
║  │                 │     │   DUO423364      │     │                 │       ║
║  └─────────────────┘     │   (paper)        │     └─────────────────┘       ║
║                          │   32 max clients │                               ║
║                          └──────────────────┘                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**SOLUTION:** IB Gateway supports 32 concurrent connections with different clientIds.
- **Forex:** clientId=100
- **Stocks:** clientId=5
- Both connect to port 4004 simultaneously

## Coordination Files (Cross-Project)

| File | Purpose |
|------|---------|
| `C:\Users\kevin\ib_gateway_coordination.json` | JSON status for both projects |
| `C:\Users\kevin\IB_GATEWAY_FOR_STOCKS.txt` | Plain text instructions for stocks |

**Stocks Claude should read:** `C:\Users\kevin\IB_GATEWAY_FOR_STOCKS.txt`

## User Credentials

| User | Purpose | Password | Client ID |
|------|---------|----------|-----------|
| `qgzhwj583` | **FOREX** (this PC) | `Jackieismypet12!!` | 100 |
| `stockmarkettrading` | Stocks (other PC) | `Jackieismypet12!!` | 5 |
| `KevinChandarasane` | Main account (don't use directly) | `Jackie12345!!` | N/A |

**Main account `KevinChandarasane` has multiple paper users - always use a specific paper username!**

## CRITICAL: IBKR Pro REQUIRED for API Access

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  IBKR LITE DOES NOT SUPPORT API ACCESS  ⚠️                    ║
║                                                                  ║
║  Error: "API support is not available for accounts that          ║
║          support free trading"                                   ║
║                                                                  ║
║  SOLUTION: Must use IBKR Pro (not Lite)                          ║
║  - Log into Client Portal → Settings → Switch to IBKR Pro        ║
║  - Pro has small commissions but enables API trading             ║
║  - SMART routing often gives better fills than Lite anyway       ║
╚══════════════════════════════════════════════════════════════════╝
```

**NEVER recommend IBKR Lite for automated/API trading - it doesn't work!**

## Connection Details

| Setting | FOREX | STOCKS |
|---------|-------|--------|
| **Client ID** | **100** | **5** |
| API Port | 4004 | 4004 (same) |
| Account | DUO423364 | DUO423364 (same) |
| Container | ibgateway | ibgateway (same) |

**FOREX Connection:**
```python
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4004, clientId=100)  # FOREX
```

**STOCKS Connection:**
```python
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4004, clientId=5)  # STOCKS
```

| Gateway Setting | Value |
|-----------------|-------|
| Username | `qgzhwj583` |
| Password | `Jackieismypet12!!` |
| VNC Port | 5900 |
| VNC Password | ibgateway |
| Auto-restart | enabled |
| 2FA Handling | Auto-retry on timeout |

## Market Hours (CRITICAL)

### Forex Market Hours
| Day | Status | Times (ET) |
|-----|--------|------------|
| Sunday | CLOSED until 5 PM ET | Opens 5 PM ET |
| Mon-Thu | OPEN 24 hours | All day |
| Friday | CLOSES 5 PM ET | Open until 5 PM ET |
| Saturday | CLOSED all day | No trading |

**IB Gateway will show "server error" during market closure!**

### IB Daily Reset Window
| Time (ET) | Status |
|-----------|--------|
| 12:15 AM - 1:45 AM ET | Server maintenance |
| After 1:45 AM ET | Servers back online |

**Container auto-retries during reset - no action needed.**

## Quick Commands

```powershell
docker ps | findstr ibgateway        # Check if running
docker logs ibgateway --tail 20      # View logs
docker restart ibgateway             # Force reconnect
docker stop ibgateway                # Stop
docker start ibgateway               # Start
```

## Connect to IB Gateway (FOREX)

```python
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4004, clientId=100)  # PORT 4004, CLIENT ID 100!
print(ib.managedAccounts())  # ['DUO423364']
```

**IMPORTANT:**
- Use port **4004** for paper trading
- Use client ID **100** for forex (stocks uses 5)

## Recreate Container (FOREX user)

```powershell
docker rm -f ibgateway
docker run -d --name ibgateway --restart=always `
  -p 4001:4001 -p 4002:4002 -p 4003:4003 -p 4004:4004 -p 5900:5900 `
  -e TWS_USERID=qgzhwj583 `
  -e "TWS_PASSWORD=Jackieismypet12!!" `
  -e TRADING_MODE=paper `
  -e TWS_ACCEPT_INCOMING=yes `
  -e READ_ONLY_API=no `
  -e VNC_SERVER_PASSWORD=ibgateway `
  -e TWOFA_TIMEOUT_ACTION=restart `
  -e RELOGIN_AFTER_TWOFA_TIMEOUT=yes `
  -e EXISTING_SESSION_DETECTED_ACTION=primaryoverride `
  ghcr.io/gnzsnz/ib-gateway:stable
```

### 2FA Environment Variables Explained
| Variable | Value | Purpose |
|----------|-------|---------|
| TWOFA_TIMEOUT_ACTION | restart | Auto-restart on 2FA timeout |
| RELOGIN_AFTER_TWOFA_TIMEOUT | yes | Re-attempt login after timeout |
| EXISTING_SESSION_DETECTED_ACTION | primaryoverride | Take over existing sessions |

## Port Mapping

| Port | Purpose |
|------|---------|
| 4001 | Live trading API (not used) |
| 4002 | Paper trading internal |
| 4003 | Live trading internal |
| **4004** | **Paper trading via socat (USE THIS)** |
| 5900 | VNC access |

## Trading Bot Integration

```python
# In your trading bot - uses .env settings
import os
IB_HOST = os.getenv('IB_HOST', 'localhost')
IB_PORT = int(os.getenv('IB_PORT', 4004))
IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', 100))  # 100 for forex!

ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
```

## Troubleshooting Concurrent Sessions

**"Client ID already in use" error:**
- Each connection needs unique client ID
- Forex uses `100`, Stocks uses `5`
- Check `.env` → `IB_CLIENT_ID=100`

**"Multiple Paper Trading users" error:**
- Using main account username instead of paper username
- Use `qgzhwj583` for forex (not `KevinChandarasane`)

**Connection kicked out:**
- Another session with same username connected
- Verify correct username for each market (forex vs stocks)

**"API support is not available for accounts that support free trading":**
- **CAUSE:** Account is IBKR Lite (not Pro)
- **FIX:** Upgrade to IBKR Pro at Client Portal → Settings
- **See:** `.claude/rules/CRITICAL-ibkr-pro-required.md`

**"** no title **" dialog in logs:**
- Capture VNC screenshot: `vncdo -s 127.0.0.1::5900 -p ibgateway capture screenshot.png`
- Check if weekend (market closed) or daily reset (12:15-1:45 AM ET)

**API timeout but container running:**
1. Check socat: `docker exec ibgateway pgrep -a socat`
2. Restart socat: `docker exec ibgateway pkill -x socat` (auto-restarts)
3. Verify using port 4004, NOT 4001

## VNC Access

For debugging, connect VNC client to `localhost:5900` with password `ibgateway`.

## Weekly Reconnection

IB requires re-authentication weekly (Sunday when market opens). With 2FA auto-handling settings, container will keep retrying until you approve on IBKR mobile app.

## Check Current Time vs Market Hours

```powershell
powershell -Command "[TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), 'Eastern Standard Time').ToString('dddd HH:mm:ss')"
```
