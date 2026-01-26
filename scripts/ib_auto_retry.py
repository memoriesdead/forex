#!/usr/bin/env python3
"""Auto-retry IB Gateway every 30 minutes until Pro upgrade takes effect."""
import subprocess
import time
from datetime import datetime

print("=" * 50)
print("IB GATEWAY AUTO-RETRY (every 30 min)")
print("Waiting for IBKR Pro upgrade to take effect...")
print("=" * 50)

for i in range(48):  # 48 x 30 min = 24 hours
    now = datetime.now().strftime('%H:%M:%S')

    # Restart gateway
    subprocess.run(['docker', 'restart', 'ibgateway'], capture_output=True)
    time.sleep(30)

    # Check logs
    result = subprocess.run(
        ['docker', 'logs', 'ibgateway', '--tail', '50'],
        capture_output=True, text=True
    )
    logs = result.stdout + result.stderr

    if 'Second Factor' in logs or 'Authenticated' in logs or 'acceptingclients' in logs:
        print(f"\n{'='*50}")
        print(f"[{now}] *** SUCCESS! 2FA READY OR CONNECTED! ***")
        print(f"*** APPROVE 2FA ON IBKR MOBILE APP NOW! ***")
        print(f"{'='*50}")
        break
    elif 'API support is not available' in logs:
        print(f"[{now}] Check {i+1}/48: Still Lite (waiting for Pro upgrade)...")
    else:
        print(f"[{now}] Check {i+1}/48: Checking status...")

    # Wait 30 minutes before next check
    time.sleep(1770)  # 30 min - 30s startup = 29.5 min

print("\nAuto-retry complete. Check 'docker logs ibgateway' for status.")
