"""Run one exact synthetic benchmark suite against an owned pair server."""
from pathlib import Path
import argparse,json,subprocess
PAIRS={
 'exp047-control':'qwen-normal-exp047',
 'exp048-kda2':'qwen-normal-exp048',
 'exp050-control-after':'qwen-normal-exp050',
}
p=argparse.ArgumentParser();p.add_argument('--root',choices=sorted(PAIRS),required=True);p.add_argument('--model',required=True);p.add_argument('--suite',choices=['performance','quality-needles'],required=True);a=p.parse_args()
assert PAIRS[a.root]==a.model
r=Path('/home/operator/qwen38-5090/optimization-runs')/a.root
assert r.resolve()==r and r.is_dir() and not r.is_symlink()
o=json.loads((r/'server-pid.json').read_text());proc=Path('/proc')/str(o['pid'])
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id']
assert (proc/'stat').read_text().split()[21]==o['stat'].split()[21]
out=r/'results'/(a.suite+'.jsonl');assert not out.exists() and out.parent.is_dir()
cmd=['python3',str(r/'benchmark_stream.py'),'--protocol','openai','--url','http://127.0.0.1:34731','--expected-model',a.model,'--manifest',str(r/'cases'/(a.suite+'.json')),'--output',str(out)]
env={'PATH':'/usr/bin:/bin','HOME':str(r/'home'),'TMPDIR':'/tmp/qwen-'+a.root+'/tmp','PYTHONDONTWRITEBYTECODE':'1'}
q=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=1200)
print(json.dumps({'root':a.root,'suite':a.suite,'exit':q.returncode,'stderr':q.stderr}),flush=True)
q.check_returncode()
rows=[json.loads(x) for x in out.read_text().splitlines()]
cases=[x for x in rows if x.get('record_type')=='case']
expected=27 if a.suite=='performance' else 3
assert len(cases)==expected and all(x.get('status')=='ok' for x in cases)
print(json.dumps({'status':'PASS','root':a.root,'suite':a.suite,'cases':len(cases)}),flush=True)
