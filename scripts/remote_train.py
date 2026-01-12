#!/usr/bin/env python3
"""
Remote Training Executor for Oracle Cloud - Forex Trading Project
Runs ML training on Oracle Cloud instance to offload compute from local machine

WARNING: This script operates on a SHARED Oracle Cloud instance (89.168.65.47)
         - Instance is shared with 5 other projects
         - All operations RESTRICTED to /home/ubuntu/projects/forex/
         - Uses oracle_helper.py which enforces path validation
         - Project ID: forex-trading-ml

See PROJECT_REGISTRY.md on Oracle Cloud for full instance documentation.
"""

import argparse
import sys
from pathlib import Path
from oracle_helper import get_oracle_cloud


def run_remote_training(
    script_path: str,
    data_on_cloud: bool = True,
    download_model: bool = True
) -> bool:
    """
    Execute training script on Oracle Cloud

    Args:
        script_path: Path to training script (relative to project root)
        data_on_cloud: If True, assumes data is already on cloud (faster)
        download_model: If True, downloads trained model after completion
    """
    oracle = get_oracle_cloud()

    print(f"[INFO] Running training on Oracle Cloud: {script_path}")

    # Step 1: Upload training script
    print("[1/4] Uploading training script...")
    local_script = Path(script_path)
    if not local_script.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False

    remote_script_path = f"scripts/{local_script.name}"
    if not oracle.upload_file(local_script, remote_script_path):
        print("[ERROR] Failed to upload script")
        return False

    # Step 2: Upload data if needed
    if not data_on_cloud:
        print("[2/4] Data not on cloud - use 'python oracle_sync.py push data' first")
        print("[INFO] Skipping data upload (assuming data already synced)")
    else:
        print("[2/4] Using data already on cloud")

    # Step 3: Execute training
    print("[3/4] Executing training script...")
    remote_cmd = f"cd {oracle.base_path} && python3 {remote_script_path}"

    success, stdout, stderr = oracle.ssh_command(remote_cmd)

    print("\n--- Training Output ---")
    print(stdout)
    if stderr:
        print("\n--- Errors/Warnings ---")
        print(stderr)
    print("--- End Output ---\n")

    if not success:
        print("[ERROR] Training failed")
        return False

    # Step 4: Download results
    if download_model:
        print("[4/4] Downloading trained models...")
        print("[INFO] Use 'python oracle_sync.py pull models' to download models")

    print("[SUCCESS] Remote training completed")
    return True


def setup_remote_environment() -> bool:
    """Install Python dependencies on Oracle Cloud"""
    oracle = get_oracle_cloud()

    print("[INFO] Setting up Python environment on Oracle Cloud...")

    # Check if requirements.txt exists locally
    requirements = Path("requirements.txt")
    if not requirements.exists():
        print("[WARNING] No requirements.txt found")
        return True

    # Upload requirements.txt
    print("[1/3] Uploading requirements.txt...")
    if not oracle.upload_file(requirements, "requirements.txt"):
        print("[ERROR] Failed to upload requirements.txt")
        return False

    # Install dependencies
    print("[2/3] Installing Python packages (this may take a while)...")
    cmd = f"cd {oracle.base_path} && pip3 install -r requirements.txt"
    success, stdout, stderr = oracle.ssh_command(cmd)

    if not success:
        print("[ERROR] Failed to install dependencies")
        print(stderr)
        return False

    print("[3/3] Environment setup complete")
    print(stdout)
    return True


def check_remote_status() -> None:
    """Check status of Oracle Cloud instance"""
    oracle = get_oracle_cloud()

    print("[INFO] Checking Oracle Cloud status...\n")

    # System info
    success, stdout, _ = oracle.ssh_command("uname -a && python3 --version && df -h /")
    if success:
        print("System Info:")
        print(stdout)

    # Storage info
    info = oracle.get_storage_info()
    print("\nForex Project Storage:")
    for key, value in info.items():
        if key != "raw_output":
            print(f"  {key}: {value}")

    # List recent models
    models = oracle.list_remote_files("models")
    if models:
        print(f"\nModels on cloud: {len(models)}")
        for model in models[-5:]:  # Last 5 models
            print(f"  - {model}")
    else:
        print("\nNo models found on cloud")


def main():
    parser = argparse.ArgumentParser(
        description="Remote Training Executor for Oracle Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup environment (first time)
  python remote_train.py setup

  # Check Oracle Cloud status
  python remote_train.py status

  # Run training script (data already on cloud)
  python remote_train.py run training/train_model.py

  # Run training and download results
  python remote_train.py run training/train_model.py --download
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Setup command
    subparsers.add_parser("setup", help="Setup Python environment on Oracle Cloud")

    # Status command
    subparsers.add_parser("status", help="Check Oracle Cloud status")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run training script on Oracle Cloud")
    run_parser.add_argument("script", help="Path to training script")
    run_parser.add_argument("--no-data-check", action="store_true", help="Skip data availability check")
    run_parser.add_argument("--download", action="store_true", help="Download models after training")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "setup":
        success = setup_remote_environment()
        sys.exit(0 if success else 1)

    elif args.command == "status":
        check_remote_status()

    elif args.command == "run":
        success = run_remote_training(
            args.script,
            data_on_cloud=True,
            download_model=args.download
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
