from pathlib import Path
import json,urllib.request,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908/sglang-baseline');d=json.loads((r/'server-pid.json').read_text());p=Path('/proc')/str(d['pid'])
print('PID',d['pid'],'exists',p.exists(),'same_start',p.exists() and (p/'stat').read_text().split()[21]==d['stat'].split()[21])
print('BOOT',Path('/proc/sys/kernel/random/boot_id').read_text().strip())
print('UPTIME',Path('/proc/uptime').read_text().strip())
try:
 v=json.load(urllib.request.urlopen('http://127.0.0.1:34731/v1/models',timeout=5));print('MODELS',json.dumps(v))
except Exception as e:print('API',str(e))
print('LOG_TAIL',(r/'server.log').read_text()[-6000:])
print(subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv'],capture_output=True,text=True).stdout)
