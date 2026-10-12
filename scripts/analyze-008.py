from pathlib import Path
import json,subprocess,hashlib
r=Path('/home/operator/qwen38-5090/optimization-runs/exp008-dflash8/results')
rows=[]
for line in (r/'performance.jsonl').read_text().splitlines():
 d=json.loads(line)
 if d.get('record_type')=='case':
  rows.append({'case':d['case'],'repetition':d['repetition'],'status':d['status'],'metrics':d.get('metrics'),'meta':{k:(d.get('meta') or {}).get(k) for k in ['prompt_tokens','cached_tokens','completion_tokens','spec_accept_length','spec_verify_ct','finish_reason']},'text_sha256':hashlib.sha256((d.get('text') or '').encode()).hexdigest()})
print('ROWS',json.dumps(rows))
root=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/managers')
needle='Out of memory even after retracting all other requests'
for p in root.glob('*.py'):
 lines=p.read_text().splitlines()
 for i,line in enumerate(lines):
  if needle in line: print('SOURCE',p,i+1,'\n'.join(lines[max(0,i-28):i+18]))
print(subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,memory.used','--format=csv'],capture_output=True,text=True).stdout)
