from pathlib import Path
import json,subprocess,sys
r=Path('/home/operator/qwen38-5090/optimization-runs/exp015b-ninfer8')
o=json.loads((r/'server-pid.json').read_text());p=Path('/proc')/str(o['pid'])
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (p/'stat').read_text().split()[21]==o['stat'].split()[21]
base=r.parent
assert json.loads((base/'ninfer-build-014/build-status.json').read_text())['stage']=='COMPLETE'
assert json.loads((base/'ninfer-resume-013/status.json').read_text())['state']=='COMPLETE'
for name in ['quality-needles','performance']:
 cmd=['python3',str(r/'benchmark_stream.py'),'--protocol','openai','--url','http://127.0.0.1:34732','--expected-model','qwen-normal-exp015b','--manifest',str(r/'cases'/(name+'.json')),'--output',str(r/'results'/(name+'.jsonl'))]
 q=subprocess.run(cmd,env={'PATH':'/usr/bin:/bin','HOME':str(r/'home'),'TMPDIR':'/tmp/qwen-exp015b','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=900)
 print('SUITE',name,'EXIT',q.returncode,q.stderr,flush=True)
 for line in (r/'results'/(name+'.jsonl')).read_text().splitlines():
  d=json.loads(line)
  if d.get('record_type')=='case':print(json.dumps({k:d.get(k) for k in ['case','repetition','status','usage','completion_tokens','metrics','error']}),flush=True)
 q.check_returncode()
