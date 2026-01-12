import requests, os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('VAST_AI_API_KEY')

r = requests.get(f'https://console.vast.ai/api/v0/bundles/?api_key={api_key}')
data = r.json()
h100 = [o for o in data.get('offers', []) if 'H100' in o.get('gpu_name', '')]
h100.sort(key=lambda x: x['dph_total'])

print(f"Available H100: {len(h100)}")
for o in h100[:3]:
    print(f"ID: {o['id']} - ${o['dph_total']:.2f}/hr - {o.get('reliability2', 0):.0f}% reliable")
