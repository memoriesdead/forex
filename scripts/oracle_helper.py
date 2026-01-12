"""
Oracle Cloud Helper Module for Forex Trading Project
Provides Python utilities for Oracle Cloud storage integration

WARNING: This module operates on a SHARED Oracle Cloud instance (89.168.65.47)
         - Instance is shared with 5 other projects
         - All operations RESTRICTED to /home/ubuntu/projects/forex/
         - Path validation enforced in all methods
         - Project ID: forex-trading-ml

See PROJECT_REGISTRY.md on Oracle Cloud for full instance documentation.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OracleCloudStorage:
    """Helper class for Oracle Cloud storage operations"""

    # Project isolation (shared instance with 6 projects)
    PROJECT_ID = "forex-trading-ml"
    ALLOWED_PATH_PREFIX = "/home/ubuntu/projects/forex"

    def __init__(self):
        self.host = os.getenv("ORACLE_CLOUD_HOST", "89.168.65.47")
        self.user = os.getenv("ORACLE_CLOUD_USER", "ubuntu")
        self.ssh_key = Path(os.getenv("ORACLE_CLOUD_SSH_KEY", "./ssh-key-2026-01-07 (1).key"))
        self.base_path = os.getenv("ORACLE_CLOUD_BASE_PATH", "/home/ubuntu/projects/forex")

        # Resolve SSH key path
        if not self.ssh_key.is_absolute():
            self.ssh_key = Path.cwd() / self.ssh_key

    def _validate_path(self, remote_path: str) -> bool:
        """
        Validate that remote path is within forex project directory.
        Prevents accidental access to other projects on shared instance.
        """
        if not remote_path.startswith(self.ALLOWED_PATH_PREFIX):
            print(f"[SECURITY ERROR] Path '{remote_path}' outside forex project!")
            print(f"[SECURITY ERROR] Only '{self.ALLOWED_PATH_PREFIX}' allowed")
            return False
        return True

    def ssh_command(self, remote_cmd: str) -> tuple[bool, str, str]:
        """
        Execute command on Oracle Cloud via SSH

        Returns:
            (success, stdout, stderr)
        """
        cmd = [
            "ssh",
            "-i", str(self.ssh_key),
            "-o", "StrictHostKeyChecking=no",
            f"{self.user}@{self.host}",
            remote_cmd
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def upload_file(self, local_path: Path, remote_subpath: str) -> bool:
        """Upload a file to Oracle Cloud"""
        remote_path = f"{self.base_path}/{remote_subpath}"

        # Validate path for security
        if not self._validate_path(remote_path):
            return False

        cmd = [
            "scp",
            "-i", str(self.ssh_key),
            "-o", "StrictHostKeyChecking=no",
            str(local_path),
            f"{self.user}@{self.host}:{remote_path}"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def download_file(self, remote_subpath: str, local_path: Path) -> bool:
        """Download a file from Oracle Cloud"""
        remote_path = f"{self.base_path}/{remote_subpath}"

        # Validate path for security
        if not self._validate_path(remote_path):
            return False

        cmd = [
            "scp",
            "-i", str(self.ssh_key),
            "-o", "StrictHostKeyChecking=no",
            f"{self.user}@{self.host}:{remote_path}",
            str(local_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def get_storage_info(self) -> dict:
        """Get storage usage information from Oracle Cloud"""
        success, stdout, stderr = self.ssh_command(f"du -sh {self.base_path}/* && df -h /")

        if not success:
            return {"error": stderr}

        lines = stdout.strip().split("\n")
        info = {"raw_output": stdout}

        # Parse disk usage
        for line in lines:
            if "/dev/" in line:
                parts = line.split()
                if len(parts) >= 6:
                    info["total"] = parts[1]
                    info["used"] = parts[2]
                    info["available"] = parts[3]
                    info["use_percent"] = parts[4]

        return info

    def ensure_remote_dir(self, remote_subpath: str) -> bool:
        """Ensure a directory exists on Oracle Cloud"""
        remote_path = f"{self.base_path}/{remote_subpath}"

        # Validate path for security
        if not self._validate_path(remote_path):
            return False

        success, _, _ = self.ssh_command(f"mkdir -p {remote_path}")
        return success

    def list_remote_files(self, remote_subpath: str = "", pattern: str = "*") -> list[str]:
        """List files in a remote directory"""
        remote_path = f"{self.base_path}/{remote_subpath}" if remote_subpath else self.base_path

        # Validate path for security
        if not self._validate_path(remote_path):
            return []

        success, stdout, _ = self.ssh_command(f"find {remote_path} -name '{pattern}' -type f")

        if not success:
            return []

        return [line.strip() for line in stdout.split("\n") if line.strip()]


def get_oracle_cloud() -> OracleCloudStorage:
    """Get Oracle Cloud storage instance"""
    return OracleCloudStorage()


def is_oracle_available() -> bool:
    """Check if Oracle Cloud is accessible"""
    oracle = get_oracle_cloud()
    success, _, _ = oracle.ssh_command("echo 'test'")
    return success


# Convenience functions
def remote_data_path(filename: str) -> str:
    """Get full remote path for a data file"""
    oracle = get_oracle_cloud()
    return f"{oracle.base_path}/data/{filename}"


def remote_model_path(filename: str) -> str:
    """Get full remote path for a model file"""
    oracle = get_oracle_cloud()
    return f"{oracle.base_path}/models/{filename}"


if __name__ == "__main__":
    # Test Oracle Cloud connection
    print("Testing Oracle Cloud connection...")
    oracle = get_oracle_cloud()

    success, stdout, stderr = oracle.ssh_command("hostname && df -h /")
    if success:
        print("[SUCCESS] Connected to Oracle Cloud")
        print(stdout)

        info = oracle.get_storage_info()
        print("\nStorage Info:")
        for key, value in info.items():
            if key != "raw_output":
                print(f"  {key}: {value}")
    else:
        print("[ERROR] Connection failed")
        print(stderr)
