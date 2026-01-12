import requests, os, sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('VAST_AI_API_KEY')

instance_id = 29778461

r = requests.get(f'https://console.vast.ai/api/v0/instances/?api_key={api_key}')
data = r.json()

for inst in data.get('instances', []):
    if inst['id'] == instance_id:
        print(f"Instance {instance_id}:")
        print(f"  Status: {inst.get('actual_status')}")
        print(f"  SSH: {inst.get('ssh_host')}:{inst.get('ssh_port')}")
        print(f"  GPU: {inst.get('gpu_name')}")
        sys.exit(0)

print(f"Instance {instance_id} not found")
