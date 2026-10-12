from pathlib import Path
import json
r=Path('/home/operator/qwen38-5090/optimization-runs/goal-c2be2a18-20260908T2042Z')
d=json.loads((r/'download-pid.json').read_text());p=Path('/proc')/str(d['pid'])
print(json.dumps({'pid':d['pid'],'exists':p.exists(),'same_start':p.exists() and (p/'stat').read_text().split()[21]==d['stat'].split()[21],'cmd':(p/'cmdline').read_bytes().decode().split(chr(0)) if p.exists() else [],'finished':(r/'download-result.json').exists()}))
