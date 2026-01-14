# IBKR Gateway (Oracle Cloud)

## Connection Details

| Setting | Value |
|---------|-------|
| Host | 89.168.65.47 |
| Container | ibgateway |
| Account | DUO423364 (paper trading) |
| API Port | 4001 |
| VNC Port | 5900 |
| VNC Password | ibgateway |

## Quick Commands

**Check if running:**
```bash
ssh ubuntu@89.168.65.47 "docker ps | grep ibgateway"
```

**View logs:**
```bash
ssh ubuntu@89.168.65.47 "docker logs ibgateway | tail -20"
```

**Restart:**
```bash
ssh ubuntu@89.168.65.47 "docker restart ibgateway"
```

**Stop:**
```bash
ssh ubuntu@89.168.65.47 "docker stop ibgateway"
```

**Start:**
```bash
ssh ubuntu@89.168.65.47 "cd /home/ubuntu/ibgateway && docker-compose up -d"
```

## Connect Locally

**SSH tunnel for API access:**
```powershell
ssh -i "C:\Users\kevin\forex\ssh-key-2026-01-07 (1).key" -L 4001:localhost:4001 ubuntu@89.168.65.47 -N
```

Then your trading bot connects to `localhost:4001`.

**VNC access (for manual login):**
```powershell
scripts\vnc_tunnel_ibgateway.bat
```
Then use a VNC client to connect to `localhost:5900` with password `ibgateway`.

## Session Conflict Fix

If you see error code 1100 (session conflict), the docker-compose.yml has:
```yaml
EXISTING_SESSION_DETECTED_ACTION=secondary
```

This allows the gateway to connect as a secondary session.

## Config Location

```
/home/ubuntu/ibgateway/
├── docker-compose.yml
├── .env
└── data/
```

## Trading Daemon Integration

The trading daemon (`scripts/trading_daemon.py`) connects to IB Gateway:

```python
# If running locally with SSH tunnel
ib.connect('localhost', 4001, clientId=1)

# If running on Oracle Cloud directly
ib.connect('localhost', 4001, clientId=1)
```

## Firewall Ports (Already Open)

- 4001 (IB API)
- 4002 (IB API alternate)
- 5900 (VNC)
