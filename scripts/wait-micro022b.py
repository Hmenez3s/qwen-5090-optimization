from pathlib import Path
import json,time
r=Path('/home/operator/qwen38-5090/optimization-runs/micro022/b');start=time.monotonic()
while time.monotonic()-start<900:
 p=r/'results.jsonl'
 if p.exists():
  lines=p.read_text().splitlines()
  if lines:
   try:last=json.loads(lines[-1])
   except json.JSONDecodeError:last={}
   if last.get('record_type') in ['summary','error']:
    print('FINAL',json.dumps(last));break
 owner=r/'server-pid.json'
 if owner.exists() and not (Path('/proc')/str(json.loads(owner.read_text())['pid'])).exists():print('PROCESS_EXITED');break
 time.sleep(3)
if (r/'results.jsonl').exists():
 for line in (r/'results.jsonl').read_text().splitlines():
  d=json.loads(line)
  if d['record_type']=='numerical':print('NUM',d['m'],d['seed'],d['implementation'],d['passed'],d['normalized_l2_error'],d['max_abs_error'])
  if d['record_type']=='timing':print('TIME',d['m'],d['seed'],d['implementation'],d['median_ms'])
p=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/micro-022b/stderr.log');print('STDERR','\n'.join(p.read_text().splitlines()[-18:]))
