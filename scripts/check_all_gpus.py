import requests, os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('VAST_AI_API_KEY')

r = requests.get(f'https://console.vast.ai/api/v0/bundles/?api_key={api_key}')
data = r.json()

# Fast GPUs for ML training
target_gpus = ['H100', 'A100', 'RTX 4090', 'A6000']

for gpu_name in target_gpus:
    offers = [o for o in data.get('offers', []) if gpu_name in o.get('gpu_name', '')]
    offers.sort(key=lambda x: x['dph_total'])

    if offers:
        print(f"\n{gpu_name}: {len(offers)} available")
        for o in offers[:3]:
            print(f"  ID: {o['id']} - ${o['dph_total']:.2f}/hr - {o.get('num_gpus', 1)}x GPU")
    else:
        print(f"\n{gpu_name}: None available")
