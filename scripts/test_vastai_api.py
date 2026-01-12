"""
Test Vast.ai API Connection
Verify API key is valid before running training
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

VAST_API_KEY = os.getenv('VAST_AI_API_KEY')
VAST_API_BASE = 'https://console.vast.ai/api/v0'

print("="*60)
print("VAST.AI API CONNECTION TEST")
print("="*60)

if not VAST_API_KEY:
    print("\n[FAIL] No API key found in .env")
    print("\nAdd to .env file:")
    print("VAST_AI_API_KEY=your_key_here")
    exit(1)

print(f"\n[OK] API key loaded: {VAST_API_KEY[:10]}...")

# Test 1: Get user info
print("\n[TEST 1] Getting user info...")
try:
    response = requests.get(
        f'{VAST_API_BASE}/users/current/',
        params={'api_key': VAST_API_KEY}
    )

    if response.status_code == 200:
        user_data = response.json()
        print(f"[OK] Connected as user: {user_data.get('username', 'N/A')}")
        print(f"     Credit: ${user_data.get('credit', 0):.2f}")
    else:
        print(f"[FAIL] Status {response.status_code}: {response.text}")
        print("\n[ERROR] API key is INVALID")
        print("\nHow to get your Vast.ai API key:")
        print("1. Go to https://console.vast.ai/")
        print("2. Sign in to your account")
        print("3. Click Account -> API Keys")
        print("4. Copy your API key")
        print("5. Add to .env file: VAST_AI_API_KEY=your_key")
        exit(1)
except Exception as e:
    print(f"[FAIL] Connection error: {e}")
    exit(1)

# Test 2: Search for H100 instances
print("\n[TEST 2] Searching for H100 instances...")
try:
    response = requests.get(
        f'{VAST_API_BASE}/bundles',
        params={'api_key': VAST_API_KEY}
    )

    if response.status_code == 200:
        data = response.json()
        offers = data.get('offers', [])

        # Filter H100
        h100_offers = [o for o in offers if 'H100' in o.get('gpu_name', '')]

        if h100_offers:
            print(f"[OK] Found {len(h100_offers)} H100 instances available")

            # Show cheapest 3
            h100_offers.sort(key=lambda x: x.get('dph_total', 999))
            print("\nCheapest H100 instances:")
            for i, offer in enumerate(h100_offers[:3], 1):
                print(f"  {i}. ${offer['dph_total']:.2f}/hour - {offer.get('gpu_name')} - {offer.get('cpu_ram', 0)/1024:.0f}GB RAM")
        else:
            print("[WARN] No H100 instances available right now")
            print("       Try again later or search for other GPUs")
    else:
        print(f"[FAIL] Status {response.status_code}: {response.text}")
except Exception as e:
    print(f"[FAIL] Search error: {e}")

print("\n" + "="*60)
print("API CONNECTION TEST COMPLETE")
print("="*60)
print("\nIf all tests passed, you can run:")
print("  python scripts/train_all_frameworks_h100.py")
print("\nThis will:")
print("  - Rent cheapest H100 ($1-2/hour)")
print("  - Train all models (1-3 hours)")
print("  - Auto-stop instance (total cost $1-6)")
print("="*60)
