from pathlib import Path
import json
root=Path('/home/operator/qwen38-5090/optimization-runs/goal-c2be2a18-20260908T2042Z')
p=root/'download-result.json'
if p.exists():print(p.read_text())
else:
 m=json.loads((root/'download-manifest.json').read_text());dest=Path(m['destination'])
 print(json.dumps({'status':'RUNNING','files':[{'name':p.name,'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file()],'log_tail':(root/'download.log').read_text()[-4000:]}))
