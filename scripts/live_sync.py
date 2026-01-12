#!/usr/bin/env python3
"""
Live sync: Download on Oracle Cloud → Transfer to local → Delete from Oracle.
All in one command. No waiting.
"""

import paramiko
import sys
from pathlib import Path
from datetime import datetime, timedelta

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


def live_sync_date(date_str: str):
    """Download on Oracle, transfer to local, delete from Oracle. All in one."""
    ssh = get_ssh_client()

    print(f"\n[1/3] Downloading {date_str} on Oracle Cloud...")

    # Run download script on Oracle Cloud for specific date
    download_cmd = f"""cd {ORACLE_BASE} && python3 -c "
from scripts.oracle_auto_download import download_day, clean_file, CURRENCY_PAIRS, DATA_RAW, DATA_CLEANED
from datetime import datetime
from pathlib import Path

date = datetime.strptime('{date_str}', '%Y%m%d')
print(f'Downloading {{date.strftime(\\\"%Y-%m-%d\\\")}}...')

total_ticks = 0
for pair in CURRENCY_PAIRS:
    ticks = download_day(pair, date)
    if ticks > 0:
        total_ticks += ticks
        print(f'  {{pair}}: {{ticks:,}} ticks')

print(f'\\nTotal: {{total_ticks:,}} ticks')

print('\\nCleaning...')
for pair in CURRENCY_PAIRS:
    input_file = DATA_RAW / f'{{pair}}_{date_str}.csv'
    output_file = DATA_CLEANED / f'{{pair}}_{date_str}.csv'
    if input_file.exists():
        cleaned = clean_file(input_file, output_file)
        if cleaned > 0:
            print(f'  {{pair}}: {{cleaned:,}} ticks')
"
"""

    stdin, stdout, stderr = ssh.exec_command(download_cmd)

    # Stream output in real-time
    for line in stdout:
        print(line.strip())

    error = stderr.read().decode()
    if error and "Traceback" in error:
        print(f"Download failed: {error}")
        ssh.close()
        return False

    print(f"\n[2/3] Transferring to local...")

    LOCAL_DATA_RAW.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_CLEANED.mkdir(parents=True, exist_ok=True)

    sftp = ssh.open_sftp()

    oracle_data_raw = f"{ORACLE_BASE}/data/dukascopy_local"
    oracle_data_cleaned = f"{ORACLE_BASE}/data_cleaned/dukascopy_local"

    # Find files for this date
    try:
        raw_files = [f for f in sftp.listdir(oracle_data_raw) if date_str in f]
        cleaned_files = [f for f in sftp.listdir(oracle_data_cleaned) if date_str in f]
    except FileNotFoundError:
        print(f"No data found for {date_str}")
        sftp.close()
        ssh.close()
        return False

    # Transfer raw data
    for filename in raw_files:
        remote_path = f"{oracle_data_raw}/{filename}"
        local_path = LOCAL_DATA_RAW / filename

        sftp.get(remote_path, str(local_path))
        print(f"  {filename}")

    # Transfer cleaned data
    for filename in cleaned_files:
        remote_path = f"{oracle_data_cleaned}/{filename}"
        local_path = LOCAL_DATA_CLEANED / filename

        sftp.get(remote_path, str(local_path))

    print(f"\n[3/3] Deleting from Oracle Cloud (free space)...")

    # Delete from Oracle
    for filename in raw_files:
        remote_path = f"{oracle_data_raw}/{filename}"
        sftp.remove(remote_path)

    for filename in cleaned_files:
        remote_path = f"{oracle_data_cleaned}/{filename}"
        sftp.remove(remote_path)

    print(f"  Deleted {len(raw_files)} raw + {len(cleaned_files)} cleaned files")

    sftp.close()
    ssh.close()

    print(f"\nDone! {date_str} is ready locally.")
    return True


def live_sync_yesterday():
    """Download yesterday's data (most common use case)."""
    yesterday = datetime.now() - timedelta(days=1)

    # Skip weekends
    if yesterday.weekday() >= 5:
        print(f"Yesterday was weekend ({yesterday.strftime('%Y-%m-%d')}), skipping")
        return False

    date_str = yesterday.strftime('%Y%m%d')
    print(f"Live sync: Yesterday ({yesterday.strftime('%Y-%m-%d')})")

    return live_sync_date(date_str)


def main():
    if len(sys.argv) < 2:
        print("Live Sync - Download on Oracle → Transfer to local → Delete from Oracle")
        print("\nUsage:")
        print("  python scripts/live_sync.py yesterday       # Download yesterday")
        print("  python scripts/live_sync.py YYYYMMDD        # Download specific date")
        print("  python scripts/live_sync.py 2026-01-08      # Download specific date")
        print("\nNo waiting. All in one command.")
        return

    command = sys.argv[1].lower()

    if command == 'yesterday':
        live_sync_yesterday()
    else:
        # Specific date
        date_str = command.replace('-', '')

        # Validate date format
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')

            if date_obj.weekday() >= 5:
                print(f"Weekend date ({date_obj.strftime('%Y-%m-%d')}), skipping")
                return

            print(f"Live sync: {date_obj.strftime('%Y-%m-%d')}")
            live_sync_date(date_str)

        except ValueError:
            print(f"Invalid date format: {command}")
            print("Use YYYYMMDD or YYYY-MM-DD")


if __name__ == "__main__":
    main()
