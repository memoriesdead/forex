#!/usr/bin/env python3
"""Wait for IB Gateway 2FA prompt."""
import subprocess
import time
from datetime import datetime

print("=" * 50)
print("MONITORING IB GATEWAY FOR 2FA PROMPT")
print("Will check every 2 minutes...")
print("=" * 50)

for i in range(30):  # Check for up to 1 hour
    result = subprocess.run(
        ['docker', 'logs', 'ibgateway', '--tail', '50'],
        capture_output=True, text=True
    )
    logs = result.stdout + result.stderr

    now = datetime.now().strftime('%H:%M:%S')

    if 'Second Factor' in logs or 'Authenticated' in logs or 'Login successful' in logs:
        print(f"\n{'='*50}")
        print(f"[{now}] *** 2FA READY - APPROVE NOW ON IBKR APP ***")
        print(f"{'='*50}")
        break
    elif 'server error' in logs:
        print(f"[{now}] Check {i+1}: IB still in reset (server error)...")
    elif 'Connecting to server' in logs:
        print(f"[{now}] Check {i+1}: Connecting...")
    else:
        # Check for any login progress
        if 'Login' in logs:
            print(f"[{now}] Check {i+1}: Login in progress...")
        else:
            print(f"[{now}] Check {i+1}: Waiting...")

    time.sleep(120)  # 2 minutes

print("\nMonitoring complete. Check 'docker logs ibgateway' manually if needed.")
