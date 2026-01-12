"""
Sync Live Forex Data from Oracle Cloud (Version 2)

Pull latest real-time data captured on Oracle Cloud to local machine
Supports various modes: latest N minutes, specific date, continuous streaming

Usage:
    python scripts/sync_live_data_v2.py --latest 1m    # Pull last minute
    python scripts/sync_live_data_v2.py --latest 1h    # Pull last hour
    python scripts/sync_live_data_v2.py --date 2026-01-08  # Pull specific day
    python scripts/sync_live_data_v2.py --stream       # Continuous sync every 5s
    python scripts/sync_live_data_v2.py --status       # Check Oracle capture status
"""

import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ORACLE_HOST = os.getenv('ORACLE_CLOUD_HOST', '89.168.65.47')
ORACLE_USER = os.getenv('ORACLE_CLOUD_USER', 'ubuntu')
ORACLE_KEY = os.getenv('ORACLE_CLOUD_SSH_KEY', './ssh-key-2026-01-07 (1).key')
ORACLE_DATA_DIR = "/home/ubuntu/projects/forex/data/live"

LOCAL_DATA_DIR = Path("data/live")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_ssh_command(command: str) -> str:
    """Execute command on Oracle Cloud via SSH"""
    ssh_cmd = [
        'ssh',
        '-i', ORACLE_KEY,
        '-o', 'StrictHostKeyChecking=no',
        f'{ORACLE_USER}@{ORACLE_HOST}',
        command
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] SSH command failed: {e}")
        return ""


def rsync_data(remote_path: str, local_path: Path):
    """Rsync data from Oracle Cloud to local"""
    local_path.mkdir(parents=True, exist_ok=True)

    rsync_cmd = [
        'rsync',
        '-avz',
        '--progress',
        '-e', f'ssh -i {ORACLE_KEY} -o StrictHostKeyChecking=no',
        f'{ORACLE_USER}@{ORACLE_HOST}:{remote_path}',
        str(local_path)
    ]

    try:
        subprocess.run(rsync_cmd, check=True)
        print(f"[OK] Synced: {remote_path} → {local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] rsync failed: {e}")
        return False


def sync_latest_minutes(minutes: int):
    """Sync last N minutes of data"""
    print(f"\n[SYNC] Pulling last {minutes} minute(s) of data...")

    # Get today's date
    today = datetime.utcnow().strftime('%Y-%m-%d')
    remote_dir = f"{ORACLE_DATA_DIR}/{today}/"
    local_dir = LOCAL_DATA_DIR / "latest"

    # Sync entire day directory (rsync will only transfer new/changed files)
    if rsync_data(remote_dir, local_dir):
        # Filter files to only keep last N minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        for csv_file in local_dir.glob("*.csv"):
            # Read and filter
            lines_to_keep = []
            with open(csv_file, 'r') as f:
                header = f.readline()
                lines_to_keep.append(header)

                for line in f:
                    if not line.strip():
                        continue

                    # Parse timestamp (format: 2026-01-08 12:34:56.123)
                    timestamp_str = line.split(',')[0]
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        if timestamp >= cutoff_time:
                            lines_to_keep.append(line)
                    except ValueError:
                        continue

            # Rewrite file with filtered data
            with open(csv_file, 'w') as f:
                f.writelines(lines_to_keep)

            row_count = len(lines_to_keep) - 1  # Exclude header
            print(f"  {csv_file.name}: {row_count} rows")

    print(f"[OK] Latest {minutes} minute(s) available in: {local_dir}")


def sync_specific_date(date_str: str):
    """Sync all data for a specific date"""
    print(f"\n[SYNC] Pulling data for {date_str}...")

    remote_dir = f"{ORACLE_DATA_DIR}/{date_str}/"
    local_dir = LOCAL_DATA_DIR / date_str

    # Check if date exists on Oracle Cloud
    check_cmd = f"test -d {remote_dir} && echo 'exists'"
    result = run_ssh_command(check_cmd)

    if result != 'exists':
        print(f"[ERROR] No data found for {date_str} on Oracle Cloud")
        return False

    # Sync entire day
    if rsync_data(remote_dir, local_dir):
        # Count rows
        total_rows = 0
        for csv_file in local_dir.glob("*.csv"):
            with open(csv_file, 'r') as f:
                rows = sum(1 for _ in f) - 1  # Exclude header
                total_rows += rows
                print(f"  {csv_file.name}: {rows} rows")

        print(f"[OK] {date_str} data available in: {local_dir}")
        print(f"[OK] Total rows: {total_rows}")
        return True

    return False


def sync_stream(interval: int = 5):
    """Continuously sync latest data every N seconds"""
    print(f"\n[STREAM] Starting continuous sync (every {interval}s)...")
    print("[STREAM] Press Ctrl+C to stop\n")

    try:
        while True:
            sync_latest_minutes(1)  # Always pull last minute
            print(f"[STREAM] Waiting {interval}s...\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[STREAM] Stopped by user")


def check_oracle_status():
    """Check if capture script is running on Oracle Cloud"""
    print("\n[STATUS] Checking Oracle Cloud capture script...")

    # Check if process is running
    check_cmd = "ps aux | grep live_capture_truefx | grep -v grep || echo 'not_running'"
    result = run_ssh_command(check_cmd)

    if result == 'not_running':
        print("[WARNING] Capture script is NOT running on Oracle Cloud")
        print("[ACTION] Start it with: bash scripts/oracle.sh ssh 'cd /home/ubuntu/projects/forex && python3 scripts/live_capture_truefx.py &'")
        return False

    print("[OK] Capture script is running")

    # Check latest data timestamp
    today = datetime.utcnow().strftime('%Y-%m-%d')
    check_cmd = f"tail -1 {ORACLE_DATA_DIR}/{today}/EURUSD_*.csv 2>/dev/null || echo 'no_data'"
    result = run_ssh_command(check_cmd)

    if result == 'no_data':
        print("[WARNING] No data found for today")
        return False

    # Parse timestamp from last line
    try:
        timestamp_str = result.split(',')[0]
        last_update = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        age_seconds = (datetime.utcnow() - last_update).total_seconds()

        print(f"[OK] Latest data: {timestamp_str} ({age_seconds:.0f}s ago)")

        if age_seconds > 60:
            print("[WARNING] Data is stale (>60s old)")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Failed to parse latest timestamp: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Sync live forex data from Oracle Cloud')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--latest', help='Sync latest N minutes/hours (e.g., 1m, 5m, 1h)')
    group.add_argument('--date', help='Sync specific date (YYYY-MM-DD)')
    group.add_argument('--stream', action='store_true', help='Continuous sync mode')
    group.add_argument('--status', action='store_true', help='Check Oracle Cloud status')

    args = parser.parse_args()

    print("="*60)
    print("LIVE DATA SYNC - Oracle Cloud -> Local")
    print(f"Oracle: {ORACLE_USER}@{ORACLE_HOST}:{ORACLE_DATA_DIR}")
    print(f"Local: {LOCAL_DATA_DIR.absolute()}")
    print("="*60)

    if args.status:
        check_oracle_status()

    elif args.latest:
        # Parse time unit
        value = args.latest[:-1]
        unit = args.latest[-1].lower()

        if not value.isdigit():
            print("[ERROR] Invalid format. Use: 1m, 5m, 1h, etc.")
            return

        minutes = int(value)
        if unit == 'h':
            minutes *= 60
        elif unit != 'm':
            print("[ERROR] Invalid unit. Use 'm' for minutes or 'h' for hours.")
            return

        sync_latest_minutes(minutes)

    elif args.date:
        sync_specific_date(args.date)

    elif args.stream:
        sync_stream()


if __name__ == "__main__":
    main()
