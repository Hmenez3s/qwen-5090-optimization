from pathlib import Path
import json,shutil
base=Path('/home/operator/qwen38-5090/optimization-runs');template=base/'exp016-common8';old=base/'exp018-mtp3';r=base/'exp019-profile8';tmp=Path('/tmp/qwen-exp019')
o=json.loads((old/'server-pid.json').read_text());p=Path('/proc')/str(o['pid']);assert (p/'stat').read_text().split()[21]==o['stat'].split()[21] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id']
assert not r.exists() and not tmp.exists();r.mkdir();tmp.mkdir();(tmp/'tmp').mkdir()
for n in ['home','storage','crash','downloads','results','profiles']:(r/n).mkdir()
m=json.loads((template/'launch.json').read_text());m['cwd']=str(r);m['wait_for']={'pid':o['pid'],'start_ticks':o['stat'].split()[21],'boot_id':o['boot_id']}
for flag,value in [('--served-model-name','qwen-normal-exp019'),('--file-storage-path',str(r/'storage')),('--crash-dump-folder',str(r/'crash')),('--download-dir',str(r/'downloads'))]:m['argv'][m['argv'].index(flag)+1]=value
m['environment'].update(HOME=str(r/'home'),HF_HOME=str(r/'home/huggingface'),XDG_CACHE_HOME=str(r/'home/.cache'),TMPDIR=str(tmp/'tmp'))
(r/'launch.json').write_text(json.dumps(m,indent=2))
for name in ['exec-manifest.py','benchmark_stream.py']:shutil.copyfile(template/name,r/name)
shutil.copytree(template/'cases',r/'cases');print('PREPARED019',m['wait_for'])
