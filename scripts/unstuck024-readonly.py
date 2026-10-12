from pathlib import Path
import json,subprocess
b=Path('/home/operator/qwen38-5090/optimization-runs');w=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows');d={}
for name,cmd in [('listeners',['ss','-ltnp']),('tcp',['ss','-tan']),('gpu',['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader'])]:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=10);d[name]={'code':p.returncode,'out':p.stdout,'err':p.stderr}
for name in ['exp023-kda8','exp024-control8']:
 root=b/name;d[name]={'pid':(root/'server-pid.json').read_text() if (root/'server-pid.json').exists() else None,'executor':(root/'exec-manifest.py').read_text()}
 for log in ['stdout.log','stderr.log']:
  p=w/('experiment-'+name[3:])/log;d[name][log]='\n'.join(p.read_text().splitlines()[-65:]) if p.exists() else 'MISSING'
for f in ['/proc/net/tcp','/proc/net/tcp6','/proc/sys/kernel/random/boot_id']:d[f]=Path(f).read_text()
print(json.dumps(d))
