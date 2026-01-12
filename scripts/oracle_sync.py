#!/usr/bin/env python3
"""
Oracle Cloud Storage Sync Manager for Forex Trading Project
Handles bidirectional sync between local and Oracle Cloud instance

WARNING: This script operates on a SHARED Oracle Cloud instance (89.168.65.47)
         - Instance is shared with 5 other projects
         - This script is RESTRICTED to /home/ubuntu/projects/forex/
         - Path validation enforced to prevent cross-project conflicts
         - Project ID: forex-trading-ml

See PROJECT_REGISTRY.md on Oracle Cloud for full instance documentation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Literal

# Oracle Cloud Configuration
ORACLE_HOST = "89.168.65.47"
ORACLE_USER = "ubuntu"
SSH_KEY = Path(__file__).parent.parent / "ssh-key-2026-01-07 (1).key"
REMOTE_BASE = "/home/ubuntu/projects/forex"
LOCAL_BASE = Path(__file__).parent.parent

# Project ID for isolation (shared instance with 6 projects)
PROJECT_ID = "forex-trading-ml"
ALLOWED_REMOTE_PREFIX = "/home/ubuntu/projects/forex"

# Directory mappings
SYNC_DIRS = {
    "data": {"local": LOCAL_BASE / "data", "remote": f"{REMOTE_BASE}/data"},
    "data_cleaned": {"local": LOCAL_BASE / "data_cleaned", "remote": f"{REMOTE_BASE}/data_cleaned"},
    "models": {"local": LOCAL_BASE / "models", "remote": f"{REMOTE_BASE}/models"},
    "logs": {"local": LOCAL_BASE / "logs", "remote": f"{REMOTE_BASE}/logs"},
}


def validate_remote_path(path: str) -> bool:
    """
    Validate remote path is within forex project directory.
    Prevents accidental access to other projects on shared instance.
    """
    if not path.startswith(ALLOWED_REMOTE_PREFIX):
        print(f"[SECURITY ERROR] Path '{path}' is outside forex project directory!")
        print(f"[SECURITY ERROR] Only paths under '{ALLOWED_REMOTE_PREFIX}' are allowed")
        print(f"[SECURITY ERROR] This instance is shared with 5 other projects")
        return False
    return True


def run_command(cmd: list[str], description: str) -> bool:
    """Execute shell command with error handling"""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(e.stderr)
        return False


def sync_directory(
    direction: Literal["push", "pull"],
    target: str,
    delete: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Sync a directory between local and Oracle Cloud

    Args:
        direction: 'push' (local -> cloud) or 'pull' (cloud -> local)
        target: Directory name ('data', 'data_cleaned', 'models', 'logs', or 'all')
        delete: Delete files on destination that don't exist on source
        dry_run: Show what would be transferred without actually doing it
    """
    if target not in SYNC_DIRS and target != "all":
        print(f"[ERROR] Invalid target: {target}")
        print(f"Valid targets: {', '.join(SYNC_DIRS.keys())}, all")
        return False

    targets = list(SYNC_DIRS.keys()) if target == "all" else [target]

    for tgt in targets:
        config = SYNC_DIRS[tgt]
        local_path = config["local"]
        remote_path = config["remote"]

        # Validate remote path for security (shared instance)
        if not validate_remote_path(remote_path):
            print(f"[ERROR] Aborted: Path validation failed for {tgt}")
            return False

        # Ensure local directory exists
        local_path.mkdir(parents=True, exist_ok=True)

        # Build rsync command
        rsync_opts = [
            "-avz",  # archive, verbose, compress
            "--progress",
            "-e", f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no",
        ]

        if delete:
            rsync_opts.append("--delete")

        if dry_run:
            rsync_opts.append("--dry-run")

        if direction == "push":
            # Local -> Cloud
            src = f"{local_path}/"
            dst = f"{ORACLE_USER}@{ORACLE_HOST}:{remote_path}/"
            desc = f"Pushing {tgt} to Oracle Cloud"
        else:
            # Cloud -> Local
            src = f"{ORACLE_USER}@{ORACLE_HOST}:{remote_path}/"
            dst = f"{local_path}/"
            desc = f"Pulling {tgt} from Oracle Cloud"

        cmd = ["rsync"] + rsync_opts + [src, dst]

        if not run_command(cmd, desc):
            return False

    return True


def check_remote_space() -> None:
    """Check available space on Oracle Cloud"""
    cmd = [
        "ssh",
        "-i", str(SSH_KEY),
        f"{ORACLE_USER}@{ORACLE_HOST}",
        f"du -sh {REMOTE_BASE}/* 2>/dev/null && df -h /"
    ]
    run_command(cmd, "Checking Oracle Cloud storage usage")


def list_remote_files(target: str = "all") -> None:
    """List files on Oracle Cloud"""
    if target == "all":
        remote_path = REMOTE_BASE
    elif target in SYNC_DIRS:
        remote_path = SYNC_DIRS[target]["remote"]
    else:
        print(f"[ERROR] Invalid target: {target}")
        return

    cmd = [
        "ssh",
        "-i", str(SSH_KEY),
        f"{ORACLE_USER}@{ORACLE_HOST}",
        f"find {remote_path} -type f -exec ls -lh {{}} \\; | tail -50"
    ]
    run_command(cmd, f"Listing files in {target}")


def clean_local(target: str) -> bool:
    """Remove local data to free up space (after pushing to cloud)"""
    if target not in SYNC_DIRS and target != "all":
        print(f"[ERROR] Invalid target: {target}")
        return False

    response = input(f"[WARNING] This will DELETE local {target} files. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("[INFO] Cancelled.")
        return False

    targets = list(SYNC_DIRS.keys()) if target == "all" else [target]

    for tgt in targets:
        local_path = SYNC_DIRS[tgt]["local"]
        if local_path.exists():
            print(f"[INFO] Removing {local_path}...")
            for item in local_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
            print(f"[INFO] Cleaned {tgt}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Cloud Storage Sync Manager for Forex Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push all data to cloud
  python oracle_sync.py push all

  # Pull models from cloud
  python oracle_sync.py pull models

  # Dry run - see what would be pushed
  python oracle_sync.py push data --dry-run

  # Push and delete remote files not in local
  python oracle_sync.py push data_cleaned --delete

  # Check remote storage usage
  python oracle_sync.py status

  # List remote files
  python oracle_sync.py list models

  # Clean local data after pushing to cloud
  python oracle_sync.py clean data
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Push command
    push_parser = subparsers.add_parser("push", help="Push local data to Oracle Cloud")
    push_parser.add_argument("target", choices=list(SYNC_DIRS.keys()) + ["all"], help="What to push")
    push_parser.add_argument("--delete", action="store_true", help="Delete remote files not in local")
    push_parser.add_argument("--dry-run", action="store_true", help="Show what would be transferred")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull data from Oracle Cloud")
    pull_parser.add_argument("target", choices=list(SYNC_DIRS.keys()) + ["all"], help="What to pull")
    pull_parser.add_argument("--delete", action="store_true", help="Delete local files not on cloud")
    pull_parser.add_argument("--dry-run", action="store_true", help="Show what would be transferred")

    # Status command
    subparsers.add_parser("status", help="Check Oracle Cloud storage usage")

    # List command
    list_parser = subparsers.add_parser("list", help="List remote files")
    list_parser.add_argument("target", nargs="?", default="all", choices=list(SYNC_DIRS.keys()) + ["all"])

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Remove local data (after pushing to cloud)")
    clean_parser.add_argument("target", choices=list(SYNC_DIRS.keys()) + ["all"], help="What to clean locally")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "push":
        success = sync_directory("push", args.target, args.delete, args.dry_run)
        sys.exit(0 if success else 1)

    elif args.command == "pull":
        success = sync_directory("pull", args.target, args.delete, args.dry_run)
        sys.exit(0 if success else 1)

    elif args.command == "status":
        check_remote_space()

    elif args.command == "list":
        list_remote_files(args.target)

    elif args.command == "clean":
        success = clean_local(args.target)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
