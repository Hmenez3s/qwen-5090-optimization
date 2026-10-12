"""Parent-owned preparation; no GPU execution, no server signal, no existing-file replacement."""
from pathlib import Path
import ast,base64,json
from remote import run
r=Path('/Users/operator/qwen-5090-optimization')
names=['exec-manifest.py','wait-ready024.py','benchmark_stream.py','run-common045.py','quality-api045.py','sitecustomize-prebuilt032.py']
files={n:(r/'scripts'/n).read_text() for n in names}
for n,s in files.items():ast.parse(s)
source='''from pathlib import Path
import json,base64,shutil,os,hashlib,stat
b=Path('/home/operator/qwen38-5090/optimization-runs');old=b/'pi-serving-044';r=b/'exp045-split-triton';restore=b/'pi-serving-046'
o=json.loads((old/'server-pid.json').read_text());p=Path('/proc')/str(o['pid'])
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (p/'stat').read_text().split()[21]==o['stat'].split()[21]
assert not r.exists() and not restore.exists();assert shutil.disk_usage(b).free>5*1024**3
original=json.loads((old/'launch.json').read_text());env0=original['environment']
cache_sources={'flashinfer':Path(env0['FLASHINFER_WORKSPACE_BASE'])/'.cache/flashinfer'}
for k in ['SGLANG_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR']:cache_sources[k]=Path(env0[k])
def digest_tree(root):
 out={}
 if root.exists():
  assert root.resolve()==root and root.is_dir()
  for parent,ds,fs in os.walk(root,followlinks=False):
   for name in ds+fs:
    p=Path(parent)/name;assert not p.is_symlink()
   for name in fs:
    p=Path(parent)/name;assert stat.S_ISREG(p.stat().st_mode)
    out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return out
before={k:digest_tree(p) for k,p in cache_sources.items()}
# Only inert final libraries and tuning data; never copied Ninja/build recipes
# or compiler-cache metadata containing old absolute output paths.
selected={k:{rel:h for rel,h in records.items() if (k=='flashinfer' and rel.endswith('.so')) or (k=='SGLANG_CACHE_DIR' and rel.startswith('flashinfer/autotune/') and rel.endswith('.json'))} for k,records in before.items()}
files=json.loads(base64.b64decode(PAYLOAD))
for root in [r,restore]:
 assert not Path('/tmp/qwen-'+root.name).exists(), 'TMP namespace must be new'
 root.mkdir()
 for n in ['home','cuda-cache','cute-cache','cute-dumps','storage','crash','downloads','results']:(root/n).mkdir()
 m=json.loads(json.dumps(original));m['cwd']=str(root);m.pop('wait_for',None);env=m['environment']
 env.update(HOME=str(root/'home'),HF_HOME=str(root/'home/huggingface'),XDG_CACHE_HOME=str(root/'home/.cache'),CUDA_CACHE_PATH=str(root/'cuda-cache'),CUTE_DSL_CACHE_DIR=str(root/'cute-cache'),CUTE_DSL_DUMP_DIR=str(root/'cute-dumps'),TMPDIR='/tmp/qwen-'+root.name+'/tmp',FLASHINFER_WORKSPACE_BASE=str(root/'home'))
 for k,src in cache_sources.items():
  dst=root/'home/.cache'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
  dst.mkdir()
  for rel in selected[k]:
   target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/rel,target)
  assert digest_tree(dst)==selected[k]
  for rel in selected[k]:
   a=(src/rel).stat();z=(dst/rel).stat();assert (a.st_dev,a.st_ino)!=(z.st_dev,z.st_ino) and z.st_nlink==1
  if k!='flashinfer':env[k]=str(dst)
 if root==r:
  m['port']=34731
  m['argv'][m['argv'].index('--port')+1]='34731'
  m['argv'][m['argv'].index('--served-model-name')+1]='qwen-normal-exp045'
  for flag,value in [('--decode-attention-backend','triton'),('--speculative-attention-mode','decode'),('--speculative-draft-attention-backend','flashinfer')]:
   assert flag not in m['argv'];m['argv'] += [flag,value]
  m['wait_for']={'pid':o['pid'],'start_ticks':o['stat'].split()[21],'boot_id':o['boot_id']}
 if root==r:
  shutil.copytree(b/'common-cases',root/'cases')
 for flag,val in [('--file-storage-path',str(root/'storage')),('--crash-dump-folder',str(root/'crash')),('--download-dir',str(root/'downloads'))]:m['argv'][m['argv'].index(flag)+1]=val
 for name in (files if root==r else ['exec-manifest.py','wait-ready024.py','sitecustomize-prebuilt032.py']):
  with (root/name).open('x') as f:f.write(files[name])
 (root/'sitecustomize.py').write_text(files['sitecustomize-prebuilt032.py'])
 catalog=root/'home/.cache/flashinfer/0.6.18/120f/cached_ops'
 hashes={str(p.relative_to(catalog)):hashlib.sha256(p.read_bytes()).hexdigest() for p in catalog.glob('*/*.so')}
 assert hashes
 (root/'prebuilt-aot.json').write_text(json.dumps({'directory':str(catalog),'sha256':hashes,'flashinfer_version':'0.6.18','architecture':'120f'}))
 env['PYTHONPATH']=str(root)
 with (root/'launch.json').open('x') as f:json.dump(m,f,indent=2)
assert {k:digest_tree(p) for k,p in cache_sources.items()}==before
with (r/'protected-cache-before.json').open('x') as f:json.dump({'paths':{k:str(v) for k,v in cache_sources.items()},'hashes':before},f)
print(json.dumps({'status':'PREPARED','roots':[str(r),str(restore)],'reference_pid':o['pid'],'cache_copy_bytes_per_root':sum((cache_sources[k]/rel).stat().st_size for k in selected for rel in selected[k]),'protected_cache_unchanged':True}))
'''.replace('PAYLOAD',repr(base64.b64encode(json.dumps(files).encode()).decode()))
ast.parse(source)
p=run(source,150)
(r/'evidence/prepare045.txt').write_bytes(p.stdout+p.stderr)
print(p.stdout.decode(),p.stderr.decode());p.check_returncode()
