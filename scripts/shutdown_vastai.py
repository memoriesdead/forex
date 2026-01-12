"""
Emergency Vast.ai Shutdown Script
Destroys ALL running instances to prevent costs
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

VAST_API_KEY = os.getenv('VAST_AI_API_KEY')
VAST_API_BASE = 'https://console.vast.ai/api/v0'

print("="*60)
print("VAST.AI EMERGENCY SHUTDOWN")
print("="*60)

if not VAST_API_KEY:
    print("\n[ERROR] No API key in .env file")
    exit(1)

print(f"\n[INFO] Using API key: {VAST_API_KEY[:10]}...")

# Get all instances
print("\n[STEP 1] Checking for running instances...")
try:
    response = requests.get(
        f'{VAST_API_BASE}/instances',
        params={'api_key': VAST_API_KEY}
    )

    if response.status_code == 401:
        print("[ERROR] API key is invalid or expired")
        print("\nManual shutdown required:")
        print("1. Go to https://console.vast.ai/instances/")
        print("2. Log in to your account")
        print("3. Click 'Destroy' on any running instances")
        print("4. Confirm you have $0 charges pending")
        exit(1)

    if response.status_code != 200:
        print(f"[ERROR] API call failed: {response.status_code}")
        print(response.text)
        exit(1)

    data = response.json()
    instances = data.get('instances', [])

    if not instances:
        print("[OK] No running instances found")
        print("[OK] You are NOT being charged")
        exit(0)

    print(f"[WARNING] Found {len(instances)} instance(s) running:")

    # Show all instances
    for inst in instances:
        inst_id = inst.get('id')
        gpu_name = inst.get('gpu_name', 'Unknown')
        dph = inst.get('dph_total', 0)
        status = inst.get('actual_status', 'unknown')

        print(f"\n  Instance ID: {inst_id}")
        print(f"  GPU: {gpu_name}")
        print(f"  Cost: ${dph:.2f}/hour")
        print(f"  Status: {status}")

    # Destroy all
    print(f"\n[STEP 2] Destroying {len(instances)} instance(s)...")

    for inst in instances:
        inst_id = inst.get('id')

        try:
            destroy_response = requests.delete(
                f'{VAST_API_BASE}/instances/{inst_id}/',
                params={'api_key': VAST_API_KEY}
            )

            if destroy_response.status_code == 200:
                print(f"[OK] Destroyed instance {inst_id}")
            else:
                print(f"[ERROR] Failed to destroy {inst_id}: {destroy_response.text}")
        except Exception as e:
            print(f"[ERROR] Exception destroying {inst_id}: {e}")

    print("\n[COMPLETE] All instances shutdown request sent")
    print("[ACTION] Verify at: https://console.vast.ai/instances/")

except Exception as e:
    print(f"[ERROR] Failed to check instances: {e}")
    print("\nManual shutdown required:")
    print("1. Go to https://console.vast.ai/instances/")
    print("2. Destroy all running instances")
    exit(1)

print("\n" + "="*60)
print("SHUTDOWN COMPLETE")
print("="*60)
