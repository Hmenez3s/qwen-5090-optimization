from pathlib import Path
import json,time,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/micro027');started=time.monotonic()
while time.monotonic()-started<1450:
 if (r/'job-status.json').exists():break
 if (r/'server-pid.json').exists():
  o=json.loads((r/'server-pid.json').read_text());p=Path('/proc')/str(o['pid'])
  if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=o['boot_id'] or not p.exists():break
 elif time.monotonic()-started>30:break
 time.sleep(3)
d={}
for name in ['counter-status.json','job-status.json','counter-stdout.log','counter-stderr.log','micro-stdout.log','micro-stderr.log','results.jsonl']:
 p=r/name
 if p.exists():d[name]=p.read_text()[-12000:]
w=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/micro027/stderr.log');d['controller-stderr']=w.read_text()[-5000:]
d['gpu']=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,power.limit','--format=csv,noheader'],text=True)
print(json.dumps(d))
