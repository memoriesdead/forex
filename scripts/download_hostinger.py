#!/usr/bin/env python3
"""Download latest forex data from Hostinger SSH server."""

import os
import paramiko
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env_hostinger")

def download_from_hostinger():
    """Download all recent forex data from Hostinger via SFTP."""
    host = os.getenv("HOSTINGER_FTP_HOST")
    user = os.getenv("HOSTINGER_FTP_USER")
    password = os.getenv("HOSTINGER_FTP_PASS")
    remote_base_dir = "/root/forex_data_node"

    local_dir = Path(__file__).parent.parent / "data" / "hostinger_downloads" / "forex_data_node"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {host} via SFTP...")

    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=user, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            sftp.chdir(remote_base_dir)
        except IOError:
            print(f"Remote directory {remote_base_dir} not found, trying root...")
            sftp.chdir("/")
            files = sftp.listdir(".")
            print(f"Available directories: {files}")
            return

        files = sftp.listdir(".")
        csv_files = [f for f in files if f.endswith('.csv')]

        print(f"Found {len(csv_files)} CSV files on server")

        downloaded = 0
        skipped = 0

        for filename in csv_files:
            local_file = local_dir / filename

            try:
                remote_stat = sftp.stat(filename)
                remote_mtime = remote_stat.st_mtime

                if local_file.exists():
                    local_mtime = local_file.stat().st_mtime
                    if remote_mtime <= local_mtime:
                        skipped += 1
                        continue

                print(f"Downloading {filename}...")
                sftp.get(filename, str(local_file))
                os.utime(local_file, (remote_mtime, remote_mtime))
                downloaded += 1

            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                continue

        sftp.close()
        transport.close()

        print(f"\nDownload complete! Downloaded: {downloaded}, Skipped: {skipped}")

    except Exception as e:
        print(f"Connection error: {e}")
        print("\nTrying alternative connection methods...")

if __name__ == "__main__":
    download_from_hostinger()
