#!/usr/bin/env python3
"""Initialize download tracker with existing data on Oracle Cloud."""

import paramiko
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
ORACLE_KEY = "ssh-key-2026-01-07 (1).key"
ORACLE_BASE = "/home/ubuntu/projects/forex"

LOCAL_BASE = Path(__file__).parent.parent


def init_tracker():
    """Scan Oracle Cloud for existing data and create tracker."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key_path = LOCAL_BASE / ORACLE_KEY
    ssh.connect(ORACLE_HOST, username=ORACLE_USER, key_filename=str(key_path))

    sftp = ssh.open_sftp()

    oracle_data_raw = f"{ORACLE_BASE}/data/dukascopy_local"

    # List all files
    try:
        files = sftp.listdir(oracle_data_raw)
    except FileNotFoundError:
        print("No data directory found on Oracle Cloud")
        sftp.close()
        ssh.close()
        return

    # Group by date
    dates = defaultdict(list)
    for filename in files:
        if filename.endswith('.csv'):
            # Extract pair and date from filename (e.g., EURUSD_20260107.csv)
            parts = filename.replace('.csv', '').split('_')
            if len(parts) == 2:
                pair = parts[0]
                date_str = parts[1]
                dates[date_str].append(pair)

    # Create tracker
    tracker = {}
    for date_str, pairs in sorted(dates.items()):
        # Convert YYYYMMDD to YYYY-MM-DD
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        tracker[date_str] = {
            'date': date_obj.strftime('%Y-%m-%d'),
            'pairs': sorted(pairs),
            'downloaded_at': datetime.now().isoformat(),
            'status': 'available'
        }

    # Upload tracker
    tracker_path = f"{ORACLE_BASE}/download_tracker.json"

    with sftp.file(tracker_path, 'w') as f:
        json.dump(tracker, f, indent=2)

    print(f"Initialized tracker with {len(tracker)} dates:")
    for date_str, info in sorted(tracker.items()):
        print(f"  {info['date']}: {len(info['pairs'])} pairs")

    sftp.close()
    ssh.close()


if __name__ == "__main__":
    init_tracker()
