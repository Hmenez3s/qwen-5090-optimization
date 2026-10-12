"""Prepare final KDA2+minp service from baseline052; no signal or GPU execution."""
from pathlib import Path
import ast,base64,json
from remote import run
r=Path('/Users/operator/qwen-5090-optimization')
names=['exec-manifest.py','wait-ready024.py','benchmark_stream.py','run-final053.py','quality-final053.py','dflash_minp035.py','sitecustomize-prebuilt032.py','sitecustomize-final053.py']
files={n:(r/'scripts'/n).read_text() for n in names}
for source in files.values():ast.parse(source)
remote='''from pathlib import Path
import base64,hashlib,json,os,shutil,stat
b=Path('/home/operator/qwen38-5090/optimization-runs');live=b/'exp050-control-after';baseline=b/'pi-serving-052';root=b/'pi-serving-053'
o=json.loads((live/'server-pid.json').read_text());proc=Path('/proc')/str(o['pid']);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (proc/'stat').read_text().split()[21]==o['stat'].split()[21]
assert baseline.is_dir() and not root.exists() and not Path('/tmp/qwen-pi-serving-053').exists();assert shutil.disk_usage(b).free>5*1024**3
m=json.loads((baseline/'launch.json').read_text());assert m['port']==8081 and m['argv'][m['argv'].index('--served-model-name')+1]=='qwen3.8-27b-normal';env0=m['environment'];case_source=b/'common-cases';assert case_source.is_dir()
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
root.mkdir()
for name in ['home','cuda-cache','cute-cache','cute-dumps','storage','crash','downloads','results']:(root/name).mkdir()
m['cwd']=str(root);m['wait_for']={'pid':o['pid'],'start_ticks':o['stat'].split()[21],'boot_id':o['boot_id']};env=m['environment'];env.update(HOME=str(root/'home'),HF_HOME=str(root/'home/huggingface'),XDG_CACHE_HOME=str(root/'home/.cache'),CUDA_CACHE_PATH=str(root/'cuda-cache'),CUTE_DSL_CACHE_DIR=str(root/'cute-cache'),CUTE_DSL_DUMP_DIR=str(root/'cute-dumps'),TMPDIR='/tmp/qwen-pi-serving-053/tmp',FLASHINFER_WORKSPACE_BASE=str(root/'home'))
for key,src in cache_sources.items():
 dst=root/'home/.cache'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.mkdir()
 for rel in selected[key]:
  target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/rel,target)
 assert digest_tree(dst)==selected[key]
 for rel in selected[key]:
  a=(src/rel).stat();z=(dst/rel).stat();assert (a.st_dev,a.st_ino)!=(z.st_dev,z.st_ino) and z.st_nlink==1
 if key!='flashinfer':env[key]=str(dst)
shutil.copytree(case_source,root/'cases')
for name,source in files.items():(root/name).write_text(source)
for flag,val in [('--file-storage-path',str(root/'storage')),('--crash-dump-folder',str(root/'crash')),('--download-dir',str(root/'downloads'))]:m['argv'][m['argv'].index(flag)+1]=val
(root/'sitecustomize.py').write_text(files['sitecustomize-final053.py']);catalog=root/'home/.cache/flashinfer/0.6.18/120f/cached_ops';hashes={str(p.relative_to(catalog)):hashlib.sha256(p.read_bytes()).hexdigest() for p in catalog.glob('*/*.so')};assert hashes;(root/'prebuilt-aot.json').write_text(json.dumps({'directory':str(catalog),'sha256':hashes,'flashinfer_version':'0.6.18','architecture':'120f'}));env['PYTHONPATH']=str(root);(root/'launch.json').write_text(json.dumps(m,indent=2));(root/'protected-cache-before.json').write_text(json.dumps({'paths':{k:str(v) for k,v in cache_sources.items()},'hashes':before}));assert {key:digest_tree(path) for key,path in cache_sources.items()}==before
print(json.dumps({'status':'PREPARED','root':str(root),'wait_for_pid':o['pid'],'protected_cache_unchanged':True,'aot_libraries':len(hashes)}))
'''.replace('PAYLOAD',repr(base64.b64encode(json.dumps(files).encode()).decode()))
ast.parse(remote);p=run(remote,180);(r/'evidence/prepare053.txt').write_bytes(p.stdout+p.stderr);print(p.stdout.decode(),p.stderr.decode());p.check_returncode()
