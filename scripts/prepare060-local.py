"""Prepare an isolated DSpark NVFP4 trial from the immutable final template."""
from pathlib import Path
import ast, base64, json
from remote import run

LOCAL = Path('/Users/operator/qwen-5090-optimization')
source_names = ['benchmark_stream.py', 'exec-manifest.py', 'wait-ready024.py']
files = {name: (LOCAL / 'scripts' / name).read_text() for name in source_names}
for source in files.values():
    ast.parse(source)

remote_source = r'''from pathlib import Path
import base64,hashlib,json,os,shutil,stat
base=Path('/home/operator/qwen38-5090/optimization-runs')
template=base/'pi-serving-059'
root=base/'exp060-dspark'
draft=Path('/home/operator/qwen38-5090/models/Qwen3.8-27B-DSpark-NVFP4-eba1ac5')
case_source=base/'exp048-kda2'/'cases'
assert not root.exists() and not root.is_symlink()
assert not Path('/tmp/qwen-exp060').exists()
assert shutil.disk_usage(base).free > 6*1024**3
manifest=json.loads((draft/'download-manifest.json').read_text())
assert manifest['revision']=='eba1ac5a66c74902eaa95a4000a7c5eda96d8e95'
assert manifest['files'][-1]['sha256']=='212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05'
assert hashlib.sha256((draft/'model.safetensors').read_bytes()).hexdigest()=='212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05'
config=json.loads((draft/'config.json').read_text())
assert config['architectures']==['Qwen3DSparkModel'] and config['block_size']==7
assert config['dflash_config']['target_layer_ids']==[4,16,28,40,52]

def regular(path):
 info=path.stat();assert stat.S_ISREG(info.st_mode) and not path.is_symlink();return info

def copy_regular(src,dst):
 regular(src);assert not dst.exists() and not dst.is_symlink();dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);a=src.stat();b=dst.stat();assert (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino) and b.st_nlink==1

root.mkdir()
for name in ['home','cuda-cache','cute-cache','cute-dumps','storage','crash','downloads','results','cases']:(root/name).mkdir()
original=json.loads((template/'launch.json').read_text())
assert hashlib.sha256((template/'launch.json').read_bytes()).hexdigest()=='db3cac58ff894174d4d4eb60a3116ccfa3cb053890598d1107467562a7e651bf'
launch=json.loads(json.dumps(original));launch.pop('wait_for',None);launch['cwd']=str(root);launch['port']=34731
argv=launch['argv'];argv[argv.index('--served-model-name')+1]='qwen-normal-exp060';argv[argv.index('--port')+1]='34731';argv[argv.index('--speculative-algorithm')+1]='DSPARK';argv[argv.index('--speculative-draft-model-path')+1]=str(draft);argv[argv.index('--max-mamba-cache-size')+1]='8'
remove={'--speculative-num-steps','--speculative-eagle-topk','--speculative-num-draft-tokens','--speculative-ngram-min-bfs-breadth','--speculative-ngram-max-bfs-breadth','--speculative-ngram-max-trie-depth','--speculative-draft-window-size','--speculative-dflash-block-size'}
out=[];i=0
while i<len(argv):
 if argv[i] in remove:
  assert i+1<len(argv) and not argv[i+1].startswith('--');i+=2;continue
 out.append(argv[i]);i+=1
argv=out+['--speculative-dspark-block-size','7','--trust-remote-code'];launch['argv']=argv
for flag,value in [('--file-storage-path',str(root/'storage')),('--crash-dump-folder',str(root/'crash')),('--download-dir',str(root/'downloads'))]:argv[argv.index(flag)+1]=value
env=launch['environment']
for key,value in list(env.items()):
 if isinstance(value,str):env[key]=value.replace(str(template),str(root))
env['HOME']=str(root/'home');env['HF_HOME']=str(root/'home/huggingface');env['XDG_CACHE_HOME']=str(root/'home/.cache');env['CUDA_CACHE_PATH']=str(root/'cuda-cache');env['CUTE_DSL_CACHE_DIR']=str(root/'cute-cache');env['CUTE_DSL_DUMP_DIR']=str(root/'cute-dumps');env['TMPDIR']='/tmp/qwen-exp060/tmp';env['FLASHINFER_WORKSPACE_BASE']=str(root/'home');env['PYTHONPATH']=str(root)
for key,name in [('SGLANG_CACHE_DIR','sglang'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('TORCH_EXTENSIONS_DIR','torch-extensions')]:env[key]=str(root/'home/.cache'/name);Path(env[key]).mkdir(parents=True,exist_ok=True)

payload=json.loads(base64.b64decode(PAYLOAD))
for name,text in payload.items():(root/name).write_text(text)
for name in ['sitecustomize-prebuilt032.py','sitecustomize-kda059.py']:
 text=(template/name).read_text().replace(str(template),str(root))
 if name=='sitecustomize-kda059.py':text=text.replace('FINAL059_KDA2_ENABLED','EXP060_KDA2_ENABLED')
 (root/name).write_text(text)
(root/'sitecustomize.py').write_text((root/'sitecustomize-kda059.py').read_text())
aot=json.loads((template/'prebuilt-aot.json').read_text());src_catalog=Path(aot['directory']);dst_catalog=Path(str(src_catalog).replace(str(template),str(root),1))
for rel,digest in aot['sha256'].items():
 src=src_catalog/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==digest;copy_regular(src,dst_catalog/rel);assert hashlib.sha256((dst_catalog/rel).read_bytes()).hexdigest()==digest
aot['directory']=str(dst_catalog);(root/'prebuilt-aot.json').write_text(json.dumps(aot,indent=2))
source_cache=Path(original['environment']['SGLANG_CACHE_DIR']);dest_cache=Path(env['SGLANG_CACHE_DIR'])
if source_cache.exists():
 for src in source_cache.rglob('*'):
  if src.is_file():copy_regular(src,dest_cache/src.relative_to(source_cache))
for src in case_source.glob('*.json'):copy_regular(src,root/'cases'/src.name)
quality=(Path('/home/operator/qwen38-5090/optimization-runs/pi-serving-059/quality-api059.py').read_text().replace("ROOT=Path('/home/operator/qwen38-5090/optimization-runs/pi-serving-059')",f"ROOT=Path('{root}')").replace("base='http://127.0.0.1:8081'","base='http://127.0.0.1:34731'").replace("model='qwen3.8-27b-normal'","model='qwen-normal-exp060'"))
(root/'quality-api060.py').write_text(quality)
run="""from pathlib import Path\nimport json,subprocess\nr=Path(ROOT)\no=json.loads((r/'server-pid.json').read_text());p=Path('/proc')/str(o['pid']);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (p/'stat').read_text().split()[21]==o['stat'].split()[21]\nfor name in ['quality-needles','performance']:\n cmd=['python3',str(r/'benchmark_stream.py'),'--protocol','openai','--url','http://127.0.0.1:34731','--expected-model','qwen-normal-exp060','--manifest',str(r/'cases'/(name+'.json')),'--output',str(r/'results'/(name+'.jsonl'))]\n q=subprocess.run(cmd,env={'PATH':'/usr/bin:/bin','HOME':str(r/'home'),'TMPDIR':'/tmp/qwen-exp060/tmp','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=1200)\n print('SUITE',name,'EXIT',q.returncode,q.stderr,flush=True)\n for line in (r/'results'/(name+'.jsonl')).read_text().splitlines():\n  d=json.loads(line)\n  if d.get('record_type')=='case':print(json.dumps({k:d.get(k) for k in ['case','repetition','status','usage','completion_tokens','metrics','error']}),flush=True)\n q.check_returncode()\n""".replace('ROOT',repr(str(root)))
(root/'run-common060.py').write_text(run)
(root/'launch.json').write_text(json.dumps(launch,indent=2))
assert not (root/'server-pid.json').exists()
print(json.dumps({'status':'PREPARED','root':str(root),'draft':str(draft),'algorithm':'DSPARK','gamma':7,'context':argv[argv.index('--context-length')+1],'pool':argv[argv.index('--max-total-tokens')+1],'mamba_slots':8,'aot':len(aot['sha256']),'cases':len(list((root/'cases').glob('*.json')))}))
'''.replace('PAYLOAD',repr(base64.b64encode(json.dumps(files).encode()).decode()))
ast.parse(remote_source)
proc = run(remote_source, 300)
(LOCAL / 'evidence' / 'prepare060.txt').write_bytes(proc.stdout + proc.stderr)
print(proc.stdout.decode(), proc.stderr.decode())
proc.check_returncode()
