#!/bin/bash
# vast.ai API helpers — no CLI needed, just curl
# API key lives at ~/.vast_api_key

API_KEY=$(cat ~/.vast_api_key | tr -d '\n')
BASE="https://console.vast.ai/api/v0"

vast_search() {
  # Search for rentable GPU instances, sorted by price
  # Usage: vast_search [min_gpu_ram_gb] [max_results]
  local ram=${1:-12}
  local limit=${2:-10}
  local ram_bytes=$((ram * 1024))
  curl -sL "$BASE/bundles/?q=%7B%22gpu_ram%22%3A%7B%22gte%22%3A${ram_bytes}%7D%2C%22rentable%22%3A%7B%22eq%22%3Atrue%7D%2C%22num_gpus%22%3A%7B%22eq%22%3A1%7D%7D&order=%5B%5B%22dph_total%22%2C%22asc%22%5D%5D&limit=${limit}" \
    -H "Authorization: Bearer $API_KEY" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for o in data.get('offers', []):
    gpu = o.get('gpu_name', '?')
    ram = o.get('gpu_ram', 0) / 1024
    price = o.get('dph_total', 0)
    dl = o.get('inet_down', 0)
    up = o.get('inet_up', 0)
    rel = o.get('reliability2', 0)
    oid = o.get('id', '?')
    print(f'{oid:>8}  {gpu:<24} {ram:>5.1f}GB  \${price:.3f}/hr  ↓{dl:.0f} ↑{up:.0f} Mbps  rel:{rel:.2f}')
"
}

vast_rent() {
  # Rent an instance
  # Usage: vast_rent <offer_id> [image] [disk_gb]
  local offer_id=$1
  local image=${2:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}
  local disk=${3:-20}
  curl -sL -X PUT "$BASE/asks/$offer_id/?format=json" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"client_id\": \"me\", \"image\": \"$image\", \"disk\": $disk, \"onstart\": \"\"}"
}

vast_instances() {
  # List your running instances
  curl -sL "$BASE/instances/?format=json" \
    -H "Authorization: Bearer $API_KEY" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'instances' in data:
    for i in data['instances']:
        iid = i.get('id', '?')
        gpu = i.get('gpu_name', '?')
        status = i.get('actual_status', '?')
        ssh_host = i.get('ssh_host', '')
        ssh_port = i.get('ssh_port', '')
        price = i.get('dph_total', 0)
        print(f'{iid:>8}  {gpu:<20} {status:<12} \${price:.3f}/hr  ssh -p {ssh_port} root@{ssh_host}')
else:
    print(json.dumps(data, indent=2)[:500])
"
}

vast_destroy() {
  # Destroy an instance
  # Usage: vast_destroy <instance_id>
  curl -sL -X DELETE "$BASE/instances/$1/?format=json" \
    -H "Authorization: Bearer $API_KEY"
}

vast_ssh_cmd() {
  # Get SSH command for an instance
  # Usage: vast_ssh_cmd <instance_id>
  curl -sL "$BASE/instances/$1/?format=json" \
    -H "Authorization: Bearer $API_KEY" | python3 -c "
import json, sys
i = json.load(sys.stdin)
host = i.get('ssh_host', '')
port = i.get('ssh_port', '')
print(f'ssh -p {port} root@{host}')
"
}

# Run the function passed as first arg
"$@"
