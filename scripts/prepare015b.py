from pathlib import Path
import json,shutil
base=Path('/home/operator/qwen38-5090/optimization-runs');old=base/'exp015-ninfer8';r=base/'exp015b-ninfer8';tmp=Path('/tmp/qwen-exp015b')
o=json.loads((old/'server-pid.json').read_text());assert not (Path('/proc')/str(o['pid'])).exists()
assert not r.exists() and not tmp.exists();r.mkdir();tmp.mkdir()
for n in ['home','cache','results']:(r/n).mkdir()
m=json.loads((old/'launch.json').read_text());m.pop('wait_for');m['cwd']=str(r)
for flag,value in [('--kv-capacity','131200'),('--model-id','qwen-normal-exp015b'),('--request-log-jsonl',str(r/'requests.jsonl'))]:m['argv'][m['argv'].index(flag)+1]=value
m['environment'].update(HOME=str(r/'home'),TMPDIR=str(tmp),XDG_CACHE_HOME=str(r/'cache'),CUDA_CACHE_PATH=str(r/'cache/cuda'))
(r/'launch.json').write_text(json.dumps(m,indent=2))
for name in ['exec-manifest.py','benchmark_stream.py']:shutil.copyfile(old/name,r/name)
shutil.copytree(old/'cases',r/'cases');print('PREPARED_015B')
