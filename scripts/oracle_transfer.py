#!/usr/bin/env python3
"""
Transfer forex data from Oracle Cloud to local, then delete from Oracle.
Keeps only one dataset (local for Cursor coding).
"""

import paramiko
import json
from pathlib import Path
from datetime import datetime

ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
ORACLE_KEY = "ssh-key-2026-01-07 (1).key"
ORACLE_BASE = "/home/ubuntu/projects/forex"

LOCAL_BASE = Path(__file__).parent.parent
LOCAL_DATA_RAW = LOCAL_BASE / "data" / "dukascopy_local"
LOCAL_DATA_CLEANED = LOCAL_BASE / "data_cleaned" / "dukascopy_local"


def get_ssh_client():
    """Create SSH connection to Oracle Cloud."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key_path = LOCAL_BASE / ORACLE_KEY
    ssh.connect(ORACLE_HOST, username=ORACLE_USER, key_filename=str(key_path))

    return ssh


def get_available_dates():
    """Check what dates are available on Oracle Cloud."""
    ssh = get_ssh_client()
    sftp = ssh.open_sftp()

    tracker_path = f"{ORACLE_BASE}/download_tracker.json"

    try:
        with sftp.file(tracker_path, 'r') as f:
            tracker = json.load(f)

        available = []
        for date_str, info in tracker.items():
            if info['status'] == 'available':
                available.append({
                    'date': info['date'],
                    'date_str': date_str,
                    'pairs': info['pairs'],
                    'downloaded_at': info['downloaded_at']
                })

        sftp.close()
        ssh.close()

        return sorted(available, key=lambda x: x['date'], reverse=True)

    except FileNotFoundError:
        print("No download tracker found on Oracle Cloud")
        sftp.close()
        ssh.close()
        return []


def transfer_date(date_str: str, delete_after: bool = True):
    """Transfer data for a specific date from Oracle to local, then delete from Oracle."""
    ssh = get_ssh_client()
    sftp = ssh.open_sftp()

    LOCAL_DATA_RAW.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_CLEANED.mkdir(parents=True, exist_ok=True)

    oracle_data_raw = f"{ORACLE_BASE}/data/dukascopy_local"
    oracle_data_cleaned = f"{ORACLE_BASE}/data_cleaned/dukascopy_local"

    # Find files for this date
    try:
        raw_files = [f for f in sftp.listdir(oracle_data_raw) if date_str in f]
        cleaned_files = [f for f in sftp.listdir(oracle_data_cleaned) if date_str in f]
    except FileNotFoundError:
        print(f"No data found for {date_str} on Oracle Cloud")
        sftp.close()
        ssh.close()
        return False

    print(f"\nTransferring {len(raw_files)} raw files + {len(cleaned_files)} cleaned files for {date_str}...")

    # Transfer raw data
    for filename in raw_files:
        remote_path = f"{oracle_data_raw}/{filename}"
        local_path = LOCAL_DATA_RAW / filename

        sftp.get(remote_path, str(local_path))
        print(f"  Downloaded: {filename}")

    # Transfer cleaned data
    for filename in cleaned_files:
        remote_path = f"{oracle_data_cleaned}/{filename}"
        local_path = LOCAL_DATA_CLEANED / filename

        sftp.get(remote_path, str(local_path))
        print(f"  Downloaded: {filename} (cleaned)")

    # Delete from Oracle Cloud if requested
    if delete_after:
        print(f"\nDeleting from Oracle Cloud to free space...")

        for filename in raw_files:
            remote_path = f"{oracle_data_raw}/{filename}"
            sftp.remove(remote_path)
            print(f"  Deleted: {filename}")

        for filename in cleaned_files:
            remote_path = f"{oracle_data_cleaned}/{filename}"
            sftp.remove(remote_path)
            print(f"  Deleted: {filename} (cleaned)")

        # Update tracker status
        tracker_path = f"{ORACLE_BASE}/download_tracker.json"
        with sftp.file(tracker_path, 'r') as f:
            tracker = json.load(f)

        if date_str in tracker:
            tracker[date_str]['status'] = 'transferred'
            tracker[date_str]['transferred_at'] = datetime.now().isoformat()

        with sftp.file(tracker_path, 'w') as f:
            json.dump(tracker, f, indent=2)

    sftp.close()
    ssh.close()

    print(f"\nTransfer complete: {date_str}")
    return True


def transfer_all_available(delete_after: bool = True):
    """Transfer all available dates from Oracle to local."""
    available = get_available_dates()

    if not available:
        print("No dates available on Oracle Cloud")
        return

    print(f"\nAvailable dates on Oracle Cloud: {len(available)}")
    for item in available:
        print(f"  {item['date']}: {len(item['pairs'])} pairs")

    print(f"\nTransferring all...")

    for item in available:
        transfer_date(item['date_str'], delete_after=delete_after)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python oracle_transfer.py list              # List available dates")
        print("  python oracle_transfer.py all               # Transfer all available")
        print("  python oracle_transfer.py YYYYMMDD          # Transfer specific date")
        print("  python oracle_transfer.py all --keep        # Transfer without deleting")
        return

    command = sys.argv[1]
    delete_after = '--keep' not in sys.argv

    if command == 'list':
        available = get_available_dates()
        if available:
            print(f"\nAvailable dates on Oracle Cloud: {len(available)}")
            for item in available:
                print(f"  {item['date']}: {len(item['pairs'])} pairs (downloaded {item['downloaded_at']})")
        else:
            print("No dates available")

    elif command == 'all':
        transfer_all_available(delete_after=delete_after)

    else:
        # Assume it's a date (YYYYMMDD or YYYY-MM-DD)
        date_str = command.replace('-', '')
        transfer_date(date_str, delete_after=delete_after)


if __name__ == "__main__":
    main()
