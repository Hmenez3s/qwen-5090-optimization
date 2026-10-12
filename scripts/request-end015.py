from pathlib import Path
import json
p=Path('/home/operator/qwen38-5090/optimization-runs/exp015b-ninfer8/requests.jsonl')
for l in p.open():
 d=json.loads(l)
 if d.get('event') not in ['server_start','request_start','throughput']:
  print(json.dumps(d)[:18000]);break
