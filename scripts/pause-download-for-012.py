from pathlib import Path
import json,os,time
r=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-resume-013')
d=json.loads((r/'status.json').read_text())
if d['state']!='COMPLETE':
 owner=json.loads((r/'pid.json').read_text());p=Path('/proc')/str(owner['pid'])
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
 assert (p/'stat').read_text().split()[21]==owner['stat'].split()[21]
 pending=r/'control-pause012.next'
 with pending.open('x') as f:json.dump({'pause':True},f)
 os.replace(pending,r/'control.json')
 deadline=time.monotonic()+150
 while time.monotonic()<deadline:
  try:d=json.loads((r/'status.json').read_text())
  except json.JSONDecodeError:time.sleep(0.1);continue
  if d['state'] in ['PAUSED','COMPLETE']:break
  if d['state']=='FAILED':raise RuntimeError(d)
  time.sleep(1)
 assert d['state'] in ['PAUSED','COMPLETE'], 'pause not confirmed'
print(json.dumps(d))
