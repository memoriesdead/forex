"""
Deploy Time-Series-Library Training to Vast.ai H100
Auto-rent H100, upload data, train models, download results.
Turn $100 into highest ROI via forex trading.
"""

import subprocess
import os
from pathlib import Path
import time
from dotenv import load_dotenv
import json

load_dotenv()

VAST_API_KEY = os.getenv('VAST_AI_API_KEY')
PROJECT_ROOT = Path(__file__).parent.parent


def run_cmd(cmd):
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def rent_h100_instance():
    """Rent the cheapest available H100 instance."""
    print("\n[1/6] Searching for H100 instances...")

    # Search for H100 instances
    stdout, stderr, code = run_cmd(
        f'vastai search offers "gpu_name=H100" --api-key {VAST_API_KEY} --raw'
    )

    if code != 0:
        print(f"Error searching: {stderr}")
        return None

    # Parse and find cheapest
    offers = json.loads(stdout)
    if not offers:
        print("No H100 instances available!")
        return None

    # Sort by price
    offers.sort(key=lambda x: x.get('dph_total', 999))
    best = offers[0]

    print(f"\nFound H100: ${best['dph_total']:.2f}/hour")
    print(f"  GPU: {best['gpu_name']}")
    print(f"  RAM: {best['cpu_ram']/1024:.1f}GB")
    print(f"  Reliability: {best.get('reliability2', 0):.1f}%")

    # Rent it
    offer_id = best['id']
    print(f"\n[2/6] Renting instance {offer_id}...")

    stdout, stderr, code = run_cmd(
        f'vastai create instance {offer_id} '
        f'--image pytorch/pytorch:latest '
        f'--disk 50 '
        f'--api-key {VAST_API_KEY} '
        f'--raw'
    )

    if code != 0:
        print(f"Error renting: {stderr}")
        return None

    instance = json.loads(stdout)
    instance_id = instance['new_contract']

    print(f"Instance {instance_id} rented! Waiting for it to start...")

    # Wait for instance to be ready
    for i in range(60):
        stdout, stderr, code = run_cmd(
            f'vastai show instance {instance_id} --api-key {VAST_API_KEY} --raw'
        )

        if code == 0:
            status = json.loads(stdout)
            if status.get('actual_status') == 'running':
                ssh_host = status.get('ssh_host')
                ssh_port = status.get('ssh_port')
                print(f"\nInstance ready! SSH: {ssh_host}:{ssh_port}")
                return instance_id, ssh_host, ssh_port

        print(f"  Waiting... ({i+1}/60)")
        time.sleep(5)

    print("Timeout waiting for instance")
    return None


def upload_data_and_code(instance_id, ssh_host, ssh_port):
    """Upload training data and Time-Series-Library to H100."""
    print(f"\n[3/6] Uploading data and code to H100...")

    # Get Vast.ai SSH key
    ssh_key = Path.home() / '.ssh' / 'vastai'

    # Create tarball of data and code
    print("  Creating tarball...")
    os.chdir(PROJECT_ROOT)

    run_cmd(
        'tar -czf forex_training.tar.gz '
        'data_cleaned/dukascopy_local '
        'Time-Series-Library '
        '--exclude="*.git" '
        '--exclude="__pycache__"'
    )

    # Upload via SCP
    print("  Uploading (this may take a while)...")
    stdout, stderr, code = run_cmd(
        f'scp -i {ssh_key} -P {ssh_port} '
        f'-o StrictHostKeyChecking=no '
        f'forex_training.tar.gz root@{ssh_host}:/workspace/'
    )

    if code != 0:
        print(f"Upload failed: {stderr}")
        return False

    # Extract on H100
    print("  Extracting on H100...")
    stdout, stderr, code = run_cmd(
        f'ssh -i {ssh_key} -p {ssh_port} '
        f'-o StrictHostKeyChecking=no '
        f'root@{ssh_host} '
        f'"cd /workspace && tar -xzf forex_training.tar.gz"'
    )

    print("  Upload complete!")
    return True


def run_training(instance_id, ssh_host, ssh_port):
    """Run Time-Series-Library training on all forex pairs."""
    print(f"\n[4/6] Starting training on H100...")

    ssh_key = Path.home() / '.ssh' / 'vastai'

    # Install dependencies
    print("  Installing dependencies...")
    run_cmd(
        f'ssh -i {ssh_key} -p {ssh_port} '
        f'-o StrictHostKeyChecking=no '
        f'root@{ssh_host} '
        f'"cd /workspace/Time-Series-Library && pip install -r requirements.txt"'
    )

    # Create training script for all pairs
    training_script = """
#!/bin/bash
cd /workspace/Time-Series-Library

# Train on all 78 forex pairs
for PAIR in EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD NZDUSD EURGBP EURJPY GBPJPY; do
    echo "Training $PAIR..."
    python run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id forex_$PAIR \
        --model TimesNet \
        --data forex \
        --root_path /workspace/data_cleaned/dukascopy_local \
        --data_path ${PAIR}_*.csv \
        --features M \
        --seq_len 96 \
        --pred_len 24 \
        --train_epochs 20 \
        --batch_size 64 \
        --learning_rate 0.001 \
        --use_amp
done

echo "Training complete!"
"""

    # Upload training script
    with open('train_all_pairs.sh', 'w') as f:
        f.write(training_script)

    run_cmd(
        f'scp -i {ssh_key} -P {ssh_port} '
        f'-o StrictHostKeyChecking=no '
        f'train_all_pairs.sh root@{ssh_host}:/workspace/'
    )

    # Run training
    print("  Training started! This will take several hours...")
    print("  Monitor progress: vastai ssh {instance_id}")
    print(f"  Or SSH directly: ssh -i {ssh_key} -p {ssh_port} root@{ssh_host}")

    stdout, stderr, code = run_cmd(
        f'ssh -i {ssh_key} -p {ssh_port} '
        f'-o StrictHostKeyChecking=no '
        f'root@{ssh_host} '
        f'"bash /workspace/train_all_pairs.sh"'
    )

    print("\n  Training output:")
    print(stdout)

    if code != 0:
        print(f"\nTraining had errors: {stderr}")
        return False

    print("\n  Training complete!")
    return True


def download_models(instance_id, ssh_host, ssh_port):
    """Download trained models from H100."""
    print(f"\n[5/6] Downloading trained models...")

    ssh_key = Path.home() / '.ssh' / 'vastai'

    # Create local models directory
    models_dir = PROJECT_ROOT / 'models' / 'trained_h100'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Download checkpoints
    stdout, stderr, code = run_cmd(
        f'scp -i {ssh_key} -P {ssh_port} -r '
        f'-o StrictHostKeyChecking=no '
        f'root@{ssh_host}:/workspace/Time-Series-Library/checkpoints/* '
        f'{models_dir}/'
    )

    if code != 0:
        print(f"Download failed: {stderr}")
        return False

    print(f"  Models downloaded to: {models_dir}")
    return True


def stop_instance(instance_id):
    """Stop and destroy the H100 instance."""
    print(f"\n[6/6] Stopping H100 instance...")

    stdout, stderr, code = run_cmd(
        f'vastai destroy instance {instance_id} --api-key {VAST_API_KEY}'
    )

    if code != 0:
        print(f"Error stopping: {stderr}")
        return False

    print("  Instance stopped. Billing ended.")
    return True


def main():
    """Main deployment workflow."""
    print("="*60)
    print("VAST.AI H100 DEPLOYMENT")
    print("Time-Series-Library Training for Forex")
    print("Goal: Turn $100 into highest ROI")
    print("="*60)

    if not VAST_API_KEY:
        print("\nERROR: VAST_AI_API_KEY not found in .env")
        print("Add your API key to .env file")
        return

    # Step 1: Rent H100
    result = rent_h100_instance()
    if not result:
        print("\nFailed to rent H100 instance")
        return

    instance_id, ssh_host, ssh_port = result

    try:
        # Step 2: Upload data and code
        if not upload_data_and_code(instance_id, ssh_host, ssh_port):
            print("\nFailed to upload data")
            return

        # Step 3: Run training
        if not run_training(instance_id, ssh_host, ssh_port):
            print("\nTraining failed")
            return

        # Step 4: Download models
        if not download_models(instance_id, ssh_host, ssh_port):
            print("\nFailed to download models")
            return

        print("\n"+"="*60)
        print("SUCCESS! Models trained and downloaded.")
        print("="*60)
        print("\nNext steps:")
        print("1. Models are in: models/trained_h100/")
        print("2. Deploy to paper trading: python scripts/start_paper_trading.py")
        print("3. Validate for 2 weeks (win rate >55%, Sharpe >1.5)")
        print("4. Go live with IB API (start with $100)")
        print("="*60)

    finally:
        # Always stop instance to avoid charges
        stop_instance(instance_id)


if __name__ == "__main__":
    main()
