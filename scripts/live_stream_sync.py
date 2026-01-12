"""
Real-Time Live Data Stream from Oracle Cloud
Syncs every SECOND with PST timestamps for paper trading.
NO DELAYS. Continuous streaming.
"""

import paramiko
from pathlib import Path
import logging
from datetime import datetime, timezone
import time
import pytz

# Oracle Cloud SSH details
ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
ORACLE_KEY = Path(__file__).parent.parent / "ssh-key-2026-01-07 (1).key"

# Paths
ORACLE_LIVE_DIR = "/home/ubuntu/projects/forex/data/truefx_live"
LOCAL_LIVE_DIR = Path(__file__).parent.parent / "data" / "truefx_live"

# PST timezone
PST = pytz.timezone('America/Los_Angeles')

# Setup logging
log_file = Path(__file__).parent.parent / "logs" / "live_stream.log"
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


def convert_utc_to_pst(utc_timestamp_str: str) -> str:
    """Convert UTC timestamp to PST."""
    try:
        # Parse UTC timestamp
        utc_dt = datetime.strptime(utc_timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

        # Convert to PST
        pst_dt = utc_dt.astimezone(PST)

        return pst_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Keep 3 decimal places
    except:
        return utc_timestamp_str  # Return original if conversion fails


def process_csv_to_pst(remote_content: bytes, local_path: Path):
    """Process CSV content and convert timestamps to PST."""
    try:
        # Decode content
        lines = remote_content.decode('utf-8').strip().split('\n')

        # Process each line
        pst_lines = []
        for line in lines:
            if not line.strip():
                continue

            parts = line.split(',')
            if len(parts) >= 2:
                # First column is timestamp
                utc_timestamp = parts[0]
                pst_timestamp = convert_utc_to_pst(utc_timestamp)
                parts[0] = pst_timestamp
                pst_lines.append(','.join(parts))
            else:
                pst_lines.append(line)

        # Write to local file
        local_path.write_text('\n'.join(pst_lines) + '\n', encoding='utf-8')

    except Exception as e:
        logger.error(f"Error processing CSV: {e}")


def stream_live_data():
    """Continuously stream live data every second."""
    logger.info("=" * 60)
    logger.info("STARTING REAL-TIME LIVE DATA STREAM")
    logger.info("Sync: Every 1 SECOND")
    logger.info("Timezone: PST (America/Los_Angeles)")
    logger.info("=" * 60)

    # Create local directory
    LOCAL_LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to Oracle Cloud (keep connection alive)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=ORACLE_HOST,
            username=ORACLE_USER,
            key_filename=str(ORACLE_KEY),
            timeout=30
        )
        logger.info("Connected to Oracle Cloud")

        sftp = ssh.open_sftp()
        logger.info("SFTP session opened")

        # Get initial file list
        try:
            remote_files = [f for f in sftp.listdir(ORACLE_LIVE_DIR) if f.endswith('_live.csv')]
            logger.info(f"Monitoring {len(remote_files)} live data files")
        except FileNotFoundError:
            logger.error(f"Oracle Cloud directory not found: {ORACLE_LIVE_DIR}")
            return

        # Continuous streaming loop
        sync_count = 0
        total_bytes = 0

        logger.info("=" * 60)
        logger.info("STREAMING STARTED - Press Ctrl+C to stop")
        logger.info("=" * 60)

        while True:
            start_time = time.time()

            synced_this_round = 0

            for filename in remote_files:
                try:
                    remote_path = f"{ORACLE_LIVE_DIR}/{filename}"
                    local_path = LOCAL_LIVE_DIR / filename

                    # Get remote file
                    with sftp.open(remote_path, 'rb') as remote_file:
                        remote_content = remote_file.read()

                    # Convert timestamps to PST and write
                    process_csv_to_pst(remote_content, local_path)

                    synced_this_round += 1
                    total_bytes += len(remote_content)

                except Exception as e:
                    logger.error(f"Error syncing {filename}: {e}")

            sync_count += 1

            # Log every 10 seconds to avoid spam
            if sync_count % 10 == 0:
                logger.info(f"[{sync_count}] Synced {synced_this_round} files | {total_bytes / 1024 / 1024:.2f} MB total")

            # Sleep to maintain 1-second interval
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("STREAM STOPPED BY USER")
        logger.info(f"Total syncs: {sync_count}")
        logger.info(f"Total data: {total_bytes / 1024 / 1024:.2f} MB")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Stream failed: {e}")
        raise

    finally:
        if sftp:
            sftp.close()
        if ssh:
            ssh.disconnect()
        logger.info("Disconnected from Oracle Cloud")


if __name__ == "__main__":
    try:
        stream_live_data()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
