"""
Sync Live Data from Oracle Cloud to Local
Syncs TrueFX live capture data for paper trading access.
"""

import paramiko
from pathlib import Path
import logging
from datetime import datetime

# Oracle Cloud SSH details
ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
ORACLE_KEY = Path(__file__).parent.parent / "ssh-key-2026-01-07 (1).key"

# Paths
ORACLE_LIVE_DIR = "/home/ubuntu/projects/forex/data/truefx_live"
LOCAL_LIVE_DIR = Path(__file__).parent.parent / "data" / "truefx_live"

# Setup logging
log_file = Path(__file__).parent.parent / "logs" / "live_sync.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def sync_live_data():
    """Sync live tick data from Oracle Cloud to local."""
    logger.info("=" * 60)
    logger.info("Starting live data sync from Oracle Cloud")
    logger.info("=" * 60)

    # Create local directory
    LOCAL_LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to Oracle Cloud
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=ORACLE_HOST,
            username=ORACLE_USER,
            key_filename=str(ORACLE_KEY),
            timeout=30
        )

        sftp = ssh.open_sftp()

        # List all files on Oracle Cloud
        try:
            remote_files = sftp.listdir(ORACLE_LIVE_DIR)
        except FileNotFoundError:
            logger.error(f"Oracle Cloud directory not found: {ORACLE_LIVE_DIR}")
            return

        logger.info(f"Found {len(remote_files)} files on Oracle Cloud")

        # Get list of local files
        local_files = {f.name: f for f in LOCAL_LIVE_DIR.glob('*_live.csv')}
        logger.info(f"Found {len(local_files)} files locally")

        # Sync files
        new_files = 0
        updated_files = 0
        skipped_files = 0
        total_bytes = 0

        for filename in remote_files:
            if not filename.endswith('_live.csv'):
                continue

            remote_path = f"{ORACLE_LIVE_DIR}/{filename}"
            local_path = LOCAL_LIVE_DIR / filename

            try:
                # Get remote file stats
                remote_stat = sftp.stat(remote_path)
                remote_size = remote_stat.st_size
                remote_mtime = remote_stat.st_mtime

                # Check if we need to download
                should_download = False

                if not local_path.exists():
                    # New file
                    should_download = True
                    new_files += 1
                else:
                    # Always update live files (they're constantly growing)
                    # But only if remote is newer or larger
                    local_size = local_path.stat().st_size
                    local_mtime = local_path.stat().st_mtime

                    if remote_mtime > local_mtime or remote_size > local_size:
                        should_download = True
                        updated_files += 1
                    else:
                        skipped_files += 1

                if should_download:
                    # Download file
                    sftp.get(remote_path, str(local_path))
                    total_bytes += remote_size
                    logger.info(f"Synced: {filename} ({remote_size:,} bytes)")

            except Exception as e:
                logger.error(f"Error syncing {filename}: {e}")

        sftp.close()
        ssh.close()

        # Summary
        logger.info("=" * 60)
        logger.info("Sync complete!")
        logger.info(f"New files: {new_files}")
        logger.info(f"Updated files: {updated_files}")
        logger.info(f"Skipped (up to date): {skipped_files}")
        logger.info(f"Total downloaded: {total_bytes / 1024 / 1024:.1f} MB")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = sync_live_data()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
