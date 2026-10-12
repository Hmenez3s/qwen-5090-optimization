from pathlib import Path
import json,shutil
base=Path('/home/operator/qwen38-5090/optimization-runs');template=base/'exp016-common8';old=base/'exp023-kda8';r=base/'exp024-control8';tmp=Path('/tmp/qwen-exp024')
o=json.loads((old/'server-pid.json').read_text());p=Path('/proc')/str(o['pid']);assert (p/'stat').read_text().split()[21]==o['stat'].split()[21] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id']
assert not r.exists() and not tmp.exists();r.mkdir();tmp.mkdir();(tmp/'tmp').mkdir()
for n in ['home','storage','crash','downloads','results','profiles']:(r/n).mkdir()
m=json.loads((template/'launch.json').read_text());m['cwd']=str(r);m['wait_for']={'pid':o['pid'],'start_ticks':o['stat'].split()[21],'boot_id':o['boot_id']}
for flag,value in [('--served-model-name','qwen-normal-exp024'),('--file-storage-path',str(r/'storage')),('--crash-dump-folder',str(r/'crash')),('--download-dir',str(r/'downloads'))]:m['argv'][m['argv'].index(flag)+1]=value
m['environment'].update(HOME=str(r/'home'),HF_HOME=str(r/'home/huggingface'),XDG_CACHE_HOME=str(r/'home/.cache'),TMPDIR=str(tmp/'tmp'))
assert not m['environment'].get('PYTHONPATH'), 'unexpected baseline PYTHONPATH'
for n in ['cuda-cache','cute-cache','cute-dumps']:(r/n).mkdir()
m['environment'].update(CUDA_CACHE_PATH=str(r/'cuda-cache'),CUTE_DSL_CACHE_DIR=str(r/'cute-cache'),CUTE_DSL_DUMP_DIR=str(r/'cute-dumps'))
(r/'launch.json').write_text(json.dumps(m,indent=2))
for name in ['exec-manifest.py','benchmark_stream.py']:shutil.copyfile(template/name,r/name)
shutil.copytree(template/'cases',r/'cases');print('PREPARED024',m['wait_for'])
