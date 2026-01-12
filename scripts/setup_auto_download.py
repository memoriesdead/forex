#!/usr/bin/env python3
"""
Setup auto-download on Oracle Cloud with cron job.
Runs daily at 1 AM UTC to download previous day's forex data.
"""

import paramiko
from pathlib import Path

ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
ORACLE_KEY = "ssh-key-2026-01-07 (1).key"
ORACLE_BASE = "/home/ubuntu/projects/forex"

LOCAL_BASE = Path(__file__).parent.parent


def setup_cron():
    """Upload auto-download script and setup cron job on Oracle Cloud."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key_path = LOCAL_BASE / ORACLE_KEY
    ssh.connect(ORACLE_HOST, username=ORACLE_USER, key_filename=str(key_path))

    sftp = ssh.open_sftp()

    # Upload auto-download script
    local_script = LOCAL_BASE / "scripts" / "oracle_auto_download.py"
    remote_script = f"{ORACLE_BASE}/scripts/oracle_auto_download.py"

    print(f"Uploading auto-download script to Oracle Cloud...")
    sftp.put(str(local_script), remote_script)
    print(f"  Uploaded: {remote_script}")

    # Make script executable
    ssh.exec_command(f"chmod +x {remote_script}")

    # Create cron job (daily at 1 AM UTC)
    cron_line = f"0 1 * * * cd {ORACLE_BASE} && /usr/bin/python3 {remote_script} >> {ORACLE_BASE}/logs/auto_download.log 2>&1"

    # Check if cron job already exists
    stdin, stdout, stderr = ssh.exec_command("crontab -l 2>/dev/null")
    existing_cron = stdout.read().decode()

    if "oracle_auto_download.py" in existing_cron:
        print("\nCron job already exists:")
        for line in existing_cron.split('\n'):
            if "oracle_auto_download.py" in line:
                print(f"  {line}")
        print("\nSkipping cron setup (already configured)")
    else:
        # Add new cron job
        new_cron = existing_cron.strip() + f"\n{cron_line}\n"

        # Install new cron
        stdin, stdout, stderr = ssh.exec_command("crontab -")
        stdin.write(new_cron)
        stdin.channel.shutdown_write()

        error = stderr.read().decode()
        if error:
            print(f"Error setting up cron: {error}")
        else:
            print("\nCron job installed successfully:")
            print(f"  Schedule: Daily at 1 AM UTC")
            print(f"  Script: {remote_script}")
            print(f"  Log: {ORACLE_BASE}/logs/auto_download.log")

    # Create logs directory if needed
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
    print("  1. Auto-download runs daily at 1 AM UTC")
    print("  2. Use 'python scripts/oracle_transfer.py list' to see available dates")
    print("  3. Use 'python scripts/oracle_transfer.py all' to transfer and delete from Oracle")
    print("\nMonitor logs:")
    print(f"  ssh ubuntu@{ORACLE_HOST} 'tail -f {ORACLE_BASE}/logs/auto_download.log'")


if __name__ == "__main__":
    setup_cron()
