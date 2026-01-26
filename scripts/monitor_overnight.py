#!/usr/bin/env python3
"""
Overnight Monitoring Script
===========================
Checks IB Gateway and trading bot status every 30 minutes.
Logs to logs/overnight_monitor.log
"""

import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / 'logs' / 'overnight_monitor.log'
TRADES_LOG = PROJECT_ROOT / 'logs' / 'trades_log.csv'
CHECK_INTERVAL = 1800  # 30 minutes in seconds

def log(msg: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def check_ib_gateway():
    """Check IB Gateway container status."""
    try:
        # Check if container is running
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=ibgateway', '--format', '{{.Status}}'],
            capture_output=True, text=True, timeout=10
        )
        status = result.stdout.strip()

        if not status:
            return "STOPPED", "Container not running"

        # Check recent logs for connection status
        logs = subprocess.run(
            ['docker', 'logs', 'ibgateway', '--tail', '20'],
            capture_output=True, text=True, timeout=10
        )
        log_text = logs.stdout + logs.stderr

        if 'Login successful' in log_text or 'logged in' in log_text.lower():
            return "CONNECTED", status
        elif 'Connecting to server' in log_text:
            return "CONNECTING", status
        elif 'server error' in log_text.lower() or 'retry' in log_text.lower():
            return "RETRYING", "IB servers unavailable (weekend?)"
        else:
            return "RUNNING", status

    except Exception as e:
        return "ERROR", str(e)

def check_trading_bot():
    """Check overnight trading bot status."""
    try:
        # Count trades
        if TRADES_LOG.exists():
            with open(TRADES_LOG, 'r') as f:
                trade_count = sum(1 for _ in f) - 1  # Subtract header
        else:
            trade_count = 0

        # Check if bot process is running (look for python with overnight)
        result = subprocess.run(
            ['powershell', '-Command',
             "Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like '*overnight*'} | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True, text=True, timeout=10
        )

        # Also check for hft_trading_bot
        result2 = subprocess.run(
            ['powershell', '-Command',
             "Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True, text=True, timeout=10
        )
        python_count = int(result2.stdout.strip() or '0')

        return trade_count, python_count > 0

    except Exception as e:
        return -1, False

def check_data_feeds():
    """Check data feed status from bot output."""
    try:
        output_file = Path(r'C:\Users\kevin\AppData\Local\Temp\claude\C--Users-kevin-forex-forex\tasks\b974629.output')
        if not output_file.exists():
            return {}

        # Read last 500 lines
        result = subprocess.run(
            ['powershell', '-Command', f"Get-Content '{output_file}' -Tail 500"],
            capture_output=True, text=True, timeout=10
        )

        feeds = {
            'TrueFX': 'TrueFX' in result.stdout,
            'OANDA': 'OANDA' in result.stdout,
            'IB': 'IB Market' in result.stdout or 'IB connected' in result.stdout
        }
        return feeds

    except Exception as e:
        return {'error': str(e)}

def run_check():
    """Run a single status check."""
    log("=" * 60)
    log("OVERNIGHT MONITORING CHECK")
    log("=" * 60)

    # IB Gateway
    ib_status, ib_details = check_ib_gateway()
    log(f"IB Gateway: {ib_status} - {ib_details}")

    # Trading Bot
    trade_count, bot_running = check_trading_bot()
    log(f"Trading Bot: {'RUNNING' if bot_running else 'UNKNOWN'} - {trade_count} trades logged")

    # Data Feeds
    feeds = check_data_feeds()
    if 'error' not in feeds:
        active = [k for k, v in feeds.items() if v]
        log(f"Data Feeds: {', '.join(active) if active else 'None detected'}")

    # Market status
    now = datetime.now()
    # Forex opens Sunday 5pm EST (2pm PST)
    day = now.strftime('%A')
    hour = now.hour

    if day == 'Saturday' or (day == 'Sunday' and hour < 14):  # Before 2pm PST Sunday
        market_status = "CLOSED (Weekend)"
    elif day == 'Friday' and hour >= 14:  # After 2pm PST Friday
        market_status = "CLOSING SOON"
    else:
        market_status = "OPEN"

    log(f"Market Status: {market_status}")
    log(f"Next check in 30 minutes...")
    log("")

def main():
    """Main monitoring loop."""
    log("=" * 60)
    log("OVERNIGHT MONITOR STARTED")
    log(f"Check interval: {CHECK_INTERVAL} seconds (30 min)")
    log(f"Log file: {LOG_FILE}")
    log("=" * 60)

    while True:
        try:
            run_check()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log("Monitor stopped by user")
            break
        except Exception as e:
            log(f"ERROR: {e}")
            time.sleep(60)  # Wait 1 min on error before retry

if __name__ == '__main__':
    main()
