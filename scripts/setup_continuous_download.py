#!/usr/bin/env python3
"""
Setup continuous 24/7 download on Oracle Cloud.
Downloads ALL 85 forex pairs continuously.
"""

import paramiko
from pathlib import Path

ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
ORACLE_KEY = "ssh-key-2026-01-07 (1).key"
ORACLE_BASE = "/home/ubuntu/projects/forex"

LOCAL_BASE = Path(__file__).parent.parent


def setup_continuous():
    """Upload scripts and setup continuous download with cron."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key_path = LOCAL_BASE / ORACLE_KEY
    ssh.connect(ORACLE_HOST, username=ORACLE_USER, key_filename=str(key_path))

    sftp = ssh.open_sftp()

    print("Uploading scripts to Oracle Cloud...")

    # Upload all_forex_pairs.py
    local_pairs = LOCAL_BASE / "scripts" / "all_forex_pairs.py"
    remote_pairs = f"{ORACLE_BASE}/scripts/all_forex_pairs.py"
    sftp.put(str(local_pairs), remote_pairs)
    print(f"  Uploaded: all_forex_pairs.py (85 pairs)")

    # Upload oracle_auto_download.py
    local_download = LOCAL_BASE / "scripts" / "oracle_auto_download.py"
    remote_download = f"{ORACLE_BASE}/scripts/oracle_auto_download.py"
    sftp.put(str(local_download), remote_download)
    print(f"  Uploaded: oracle_auto_download.py (updated for ALL pairs)")

    # Make executable
    ssh.exec_command(f"chmod +x {remote_download}")
    ssh.exec_command(f"chmod +x {remote_pairs}")

    # Update cron job - run every 6 hours to catch up
    cron_jobs = [
        f"0 */6 * * * cd {ORACLE_BASE} && /usr/bin/python3 {remote_download} >> {ORACLE_BASE}/logs/auto_download.log 2>&1"
    ]

    # Get existing cron
    stdin, stdout, stderr = ssh.exec_command("crontab -l 2>/dev/null")
    existing_cron = stdout.read().decode()

    # Remove old oracle_auto_download cron jobs
    new_cron_lines = []
    for line in existing_cron.split('\n'):
        if 'oracle_auto_download.py' not in line and line.strip():
            new_cron_lines.append(line)

    # Add new cron jobs
    for job in cron_jobs:
        new_cron_lines.append(job)

    new_cron = '\n'.join(new_cron_lines) + '\n'

    # Install new cron
    stdin, stdout, stderr = ssh.exec_command("crontab -")
    stdin.write(new_cron)
    stdin.channel.shutdown_write()

    error = stderr.read().decode()
    if error:
        print(f"Error setting up cron: {error}")
    else:
        print("\nCron job updated successfully:")
        print(f"  Schedule: Every 6 hours (0, 6, 12, 18 UTC)")
        print(f"  Pairs: 85 (all major, cross, exotic)")
        print(f"  Script: {remote_download}")
        print(f"  Log: {ORACLE_BASE}/logs/auto_download.log")

    # Create logs directory
    ssh.exec_command(f"mkdir -p {ORACLE_BASE}/logs")

    # Verify current cron jobs
    stdin, stdout, stderr = ssh.exec_command("crontab -l")
    current_cron = stdout.read().decode()

    print("\nCurrent cron jobs:")
    for line in current_cron.split('\n'):
        if line.strip() and not line.startswith('#'):
            print(f"  {line}")

    sftp.close()
    ssh.close()

    print("\nSetup complete!")
    print("\nNext steps:")
    print("  1. Auto-download runs every 6 hours for ALL 85 pairs")
    print("  2. Use 'python scripts/oracle_transfer.py list' to see available dates")
    print("  3. Use 'python scripts/live_sync.py yesterday' for instant download")
    print("\nExpected data volume:")
    print("  ~85 pairs × 50k ticks/day × 2 files (raw+cleaned) = ~8.5M ticks/day")
    print("  ~127MB/day compressed data")
    print("\nMonitor:")
    print(f"  ssh ubuntu@{ORACLE_HOST} 'tail -f {ORACLE_BASE}/logs/auto_download.log'")


if __name__ == "__main__":
    setup_continuous()
