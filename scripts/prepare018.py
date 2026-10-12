from pathlib import Path
import json,shutil
base=Path('/home/operator/qwen38-5090/optimization-runs');old=base/'exp017-ninfer4';r=base/'exp018-mtp3';tmp=Path('/tmp/qwen-exp018')
o=json.loads((old/'server-pid.json').read_text());p=Path('/proc')/str(o['pid'])
assert (p/'stat').read_text().split()[21]==o['stat'].split()[21] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id']
assert not r.exists() and not tmp.exists();r.mkdir();tmp.mkdir()
for n in ['home','cache','results']:(r/n).mkdir()
m=json.loads((old/'launch.json').read_text());m['cwd']=str(r);m['wait_for']={'pid':o['pid'],'start_ticks':o['stat'].split()[21],'boot_id':o['boot_id']}
for flag,value in [('--spec','mtp'),('--draft-tokens','3'),('--model-id','qwen-normal-exp018'),('--request-log-jsonl',str(r/'requests.jsonl'))]:m['argv'][m['argv'].index(flag)+1]=value
m['environment'].update(HOME=str(r/'home'),TMPDIR=str(tmp),XDG_CACHE_HOME=str(r/'cache'),CUDA_CACHE_PATH=str(r/'cache/cuda'))
(r/'launch.json').write_text(json.dumps(m,indent=2))
for name in ['exec-manifest.py','benchmark_stream.py']:shutil.copyfile(old/name,r/name)
shutil.copytree(old/'cases',r/'cases');print('PREPARED018',m['wait_for'])
