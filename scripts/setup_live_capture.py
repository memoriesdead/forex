"""
Setup Live Tick Capture on Oracle Cloud
Installs dependencies, uploads scripts, configures systemd service.
"""

import paramiko
from pathlib import Path
import logging

# Oracle Cloud SSH details
ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
ORACLE_KEY = Path(__file__).parent.parent / "ssh-key-2026-01-07 (1).key"
ORACLE_BASE = "/home/ubuntu/projects/forex"

# Local paths
LOCAL_BASE = Path(__file__).parent.parent
LIVE_CAPTURE_SCRIPT = LOCAL_BASE / "scripts" / "truefx_live_capture.py"
SERVICE_FILE = LOCAL_BASE / "scripts" / "truefx-live-capture.service"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_live_capture():
    """Setup live capture system on Oracle Cloud."""
    logger.info("Setting up live tick capture on Oracle Cloud...")

    # Connect to Oracle Cloud
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(
        hostname=ORACLE_HOST,
        username=ORACLE_USER,
        key_filename=str(ORACLE_KEY),
        timeout=30
    )

    sftp = ssh.open_sftp()

    try:
        # 1. Install Python dependencies
        logger.info("Installing websocket-client...")
        stdin, stdout, stderr = ssh.exec_command(
            "sudo apt update && sudo apt install -y python3-websocket",
            timeout=120
        )
        stdout.channel.recv_exit_status()
        logger.info("Dependencies installed")

        # 2. Create directories
        logger.info("Creating directories...")
        ssh.exec_command(f"mkdir -p {ORACLE_BASE}/data/truefx_live")
        ssh.exec_command(f"mkdir -p {ORACLE_BASE}/logs")

        # 3. Upload live capture script
        logger.info("Uploading live capture script...")
        remote_script = f"{ORACLE_BASE}/scripts/truefx_live_capture.py"
        sftp.put(str(LIVE_CAPTURE_SCRIPT), remote_script)
        ssh.exec_command(f"chmod +x {remote_script}")
        logger.info(f"Uploaded: {remote_script}")

        # 4. Upload systemd service file
        logger.info("Uploading systemd service file...")
        sftp.put(str(SERVICE_FILE), "/tmp/truefx-live-capture.service")
        ssh.exec_command("sudo mv /tmp/truefx-live-capture.service /etc/systemd/system/")
        logger.info("Service file installed")

        # 5. Enable and start service
        logger.info("Enabling and starting service...")
        commands = [
            "sudo systemctl daemon-reload",
            "sudo systemctl enable truefx-live-capture",
            "sudo systemctl start truefx-live-capture"
        ]

        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                error = stderr.read().decode()
                logger.error(f"Command failed: {cmd}")
                logger.error(f"Error: {error}")

        # 6. Check service status
        logger.info("Checking service status...")
        stdin, stdout, stderr = ssh.exec_command("sudo systemctl status truefx-live-capture", timeout=10)
        status = stdout.read().decode()
        logger.info(f"Service status:\n{status}")

        # 7. Check logs
        logger.info("Checking initial logs...")
        stdin, stdout, stderr = ssh.exec_command(f"tail -20 {ORACLE_BASE}/logs/truefx_live.log", timeout=10)
        logs = stdout.read().decode()
        if logs:
            logger.info(f"Live capture logs:\n{logs}")

        logger.info("✓ Setup complete! Live capture is running on Oracle Cloud.")
        logger.info(f"Logs: {ORACLE_BASE}/logs/truefx_live.log")
        logger.info(f"Data: {ORACLE_BASE}/data/truefx_live/")

    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    setup_live_capture()
