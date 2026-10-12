from pathlib import Path
import json,urllib.request,time,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/exp015-ninfer8');o=json.loads((r/'server-pid.json').read_text());op=urllib.request.build_opener(urllib.request.ProxyHandler({}));start=time.monotonic()
while time.monotonic()-start<180:
 p=Path('/proc')/str(o['pid'])
 if not p.exists(): print('SERVER_EXITED');break
 try:
  with op.open('http://127.0.0.1:34732/v1/models',timeout=3) as f:d=json.load(f)
  if any(x.get('id')=='qwen-normal-exp015' for x in d.get('data',[])):print('READY',json.dumps(d));break
 except Exception:pass
 time.sleep(5)
base=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/experiment-015-ninfer8')
for name in ['stdout.log','stderr.log']:
 p=base/name;print(name,'\n'.join(p.read_text().splitlines()[-20:]))
print(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader'],text=True))
