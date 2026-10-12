from pathlib import Path
import json,shutil
base=Path('/home/operator/qwen38-5090/optimization-runs');OLD=base/'exp012-block6';ROOT=base/'exp016-common8';TEMP=Path('/tmp/qwen-exp016')
assert not ROOT.exists() and not TEMP.exists()
owner=json.loads((OLD/'server-pid.json').read_text());p=Path('/proc')/str(owner['pid'])
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
assert (p/'stat').read_text().split()[21]==owner['stat'].split()[21]
m=json.loads((OLD/'launch.json').read_text());assert m['argv'][m['argv'].index('--max-total-tokens')+1]=='131328'
ROOT.mkdir();TEMP.mkdir();(TEMP/'tmp').mkdir()
for d in ['home','storage','crash','downloads','results']:(ROOT/d).mkdir()
shutil.copytree(base/'common-cases',ROOT/'cases')
shutil.copyfile(OLD/'exec-manifest.py',ROOT/'exec-manifest.py')
shutil.copyfile(base/'client-smoke-015b/benchmark_stream.py',ROOT/'benchmark_stream.py')
for flag,value in [('--served-model-name','qwen-normal-exp016'),('--speculative-num-draft-tokens','8'),('--speculative-dflash-block-size','8'),('--file-storage-path',str(ROOT/'storage')),('--crash-dump-folder',str(ROOT/'crash')),('--download-dir',str(ROOT/'downloads'))]:m['argv'][m['argv'].index(flag)+1]=value
m['cwd']=str(ROOT);m['wait_for']={'pid':owner['pid'],'start_ticks':owner['stat'].split()[21],'boot_id':owner['boot_id']}
m['environment'].update(HOME=str(ROOT/'home'),HF_HOME=str(ROOT/'home/huggingface'),XDG_CACHE_HOME=str(ROOT/'home/.cache'),TMPDIR=str(TEMP/'tmp'))
(ROOT/'launch.json').write_text(json.dumps(m,indent=2))
print(json.dumps({'root':str(ROOT),'wait_for':m['wait_for'],'compiler_cache':m['environment']['FLASHINFER_WORKSPACE_BASE']}))
