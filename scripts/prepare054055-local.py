"""Prepare private final clone054 and identical authoritative service055; no signal/GPU run."""
from pathlib import Path
import ast,base64,json
from remote import run
r=Path('/Users/operator/qwen-5090-optimization')
names=['exec-manifest.py','wait-ready024.py','benchmark_stream.py','run-final054.py','quality-final054.py','quality-final055.py','dflash_minp035.py','sitecustomize-prebuilt032.py','sitecustomize-final054.py','sitecustomize-final055.py']
files={n:(r/'scripts'/n).read_text() for n in names}
for source in files.values():ast.parse(source)
remote='''from pathlib import Path
import base64,hashlib,json,os,shutil,stat
b=Path('/home/operator/qwen38-5090/optimization-runs');live=b/'pi-serving-053';baseline=b/'pi-serving-052';validation=b/'exp054-final-validation';final=b/'pi-serving-055'
o=json.loads((live/'server-pid.json').read_text());proc=Path('/proc')/str(o['pid']);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (proc/'stat').read_text().split()[21]==o['stat'].split()[21]
assert baseline.is_dir() and not validation.exists() and not final.exists();assert not Path('/tmp/qwen-exp054-final-validation').exists() and not Path('/tmp/qwen-pi-serving-055').exists();assert shutil.disk_usage(b).free>7*1024**3
original=json.loads((baseline/'launch.json').read_text());env0=original['environment'];case_source=b/'common-cases';assert case_source.is_dir()
cache_sources={'flashinfer':Path(env0['FLASHINFER_WORKSPACE_BASE'])/'.cache/flashinfer'}
for key in ['SGLANG_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR']:cache_sources[key]=Path(env0[key])
def digest_tree(path):
 out={}
 if path.exists():
  assert path.resolve()==path and path.is_dir()
  for parent,ds,fs in os.walk(path,followlinks=False):
   for name in ds+fs:assert not (Path(parent)/name).is_symlink()
   for name in fs:
    p=Path(parent)/name;assert stat.S_ISREG(p.stat().st_mode);out[str(p.relative_to(path))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return out
before={key:digest_tree(path) for key,path in cache_sources.items()};selected={key:{rel:h for rel,h in records.items() if (key=='flashinfer' and rel.endswith('.so')) or (key=='SGLANG_CACHE_DIR' and rel.startswith('flashinfer/autotune/') and rel.endswith('.json'))} for key,records in before.items()};files=json.loads(base64.b64decode(PAYLOAD))
for root in [validation,final]:
 root.mkdir()
 for name in ['home','cuda-cache','cute-cache','cute-dumps','storage','crash','downloads','results']:(root/name).mkdir()
 m=json.loads(json.dumps(original));m['cwd']=str(root);m.pop('wait_for',None);env=m['environment'];env.update(HOME=str(root/'home'),HF_HOME=str(root/'home/huggingface'),XDG_CACHE_HOME=str(root/'home/.cache'),CUDA_CACHE_PATH=str(root/'cuda-cache'),CUTE_DSL_CACHE_DIR=str(root/'cute-cache'),CUTE_DSL_DUMP_DIR=str(root/'cute-dumps'),TMPDIR='/tmp/qwen-'+root.name+'/tmp',FLASHINFER_WORKSPACE_BASE=str(root/'home'))
 for key,src in cache_sources.items():
  dst=root/'home/.cache'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.mkdir()
  for rel in selected[key]:
   target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/rel,target)
  assert digest_tree(dst)==selected[key]
  for rel in selected[key]:
   a=(src/rel).stat();z=(dst/rel).stat();assert (a.st_dev,a.st_ino)!=(z.st_dev,z.st_ino) and z.st_nlink==1
  if key!='flashinfer':env[key]=str(dst)
 if root==validation:
  m['port']=34731;m['argv'][m['argv'].index('--port')+1]='34731';m['argv'][m['argv'].index('--served-model-name')+1]='qwen-normal-exp054';m['wait_for']={'pid':o['pid'],'start_ticks':o['stat'].split()[21],'boot_id':o['boot_id']};shutil.copytree(case_source,root/'cases');specific=['run-final054.py','quality-final054.py','sitecustomize-final054.py'];hook='sitecustomize-final054.py'
 else:
  specific=['quality-final055.py','sitecustomize-final055.py'];hook='sitecustomize-final055.py'
 for name in ['exec-manifest.py','wait-ready024.py','benchmark_stream.py','dflash_minp035.py','sitecustomize-prebuilt032.py',*specific]:(root/name).write_text(files[name])
 for flag,val in [('--file-storage-path',str(root/'storage')),('--crash-dump-folder',str(root/'crash')),('--download-dir',str(root/'downloads'))]:m['argv'][m['argv'].index(flag)+1]=val
 (root/'sitecustomize.py').write_text(files[hook]);catalog=root/'home/.cache/flashinfer/0.6.18/120f/cached_ops';hashes={str(p.relative_to(catalog)):hashlib.sha256(p.read_bytes()).hexdigest() for p in catalog.glob('*/*.so')};assert hashes;(root/'prebuilt-aot.json').write_text(json.dumps({'directory':str(catalog),'sha256':hashes,'flashinfer_version':'0.6.18','architecture':'120f'}));env['PYTHONPATH']=str(root);(root/'launch.json').write_text(json.dumps(m,indent=2));(root/'protected-cache-before.json').write_text(json.dumps({'paths':{k:str(v) for k,v in cache_sources.items()},'hashes':before}))
assert {key:digest_tree(path) for key,path in cache_sources.items()}==before
print(json.dumps({'status':'PREPARED','roots':[str(validation),str(final)],'wait_for_pid':o['pid'],'protected_cache_unchanged':True}))
'''.replace('PAYLOAD',repr(base64.b64encode(json.dumps(files).encode()).decode()))
ast.parse(remote);p=run(remote,180);(r/'evidence/prepare054055.txt').write_bytes(p.stdout+p.stderr);print(p.stdout.decode(),p.stderr.decode());p.check_returncode()
