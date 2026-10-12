from pathlib import Path
import json,subprocess,os,shutil,socket,hashlib
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908/sglang-baseline')
m=json.loads((ROOT/'recovery-manifest.json').read_text())
assert not (ROOT/'server-pid.json').exists(), 'already launched'
for child in m['children']:
 p=Path('/proc')/str(child['pid'])
 assert not p.exists(), 'original process still present '+str(child['pid'])
with socket.socket() as sock:sock.bind(('127.0.0.1',34731))
for path,info in m['protected'].items():
 p=Path(path)
 if 'mtime_ns' in info:assert p.stat().st_mtime_ns==info['mtime_ns'] and p.stat().st_size==info['bytes'], 'protected input changed'
 if 'sha256' in info:assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'], 'protected content changed'
for name in ['home','tmp','storage','crash','downloads','torch-extensions','torchinductor','triton','cuda-cache']:(ROOT/name).mkdir()
# Copy only compiler artifacts into independent run-owned files, never KV/user data.
for source,dest in [(Path('/home/operator/.cache/flashinfer'),ROOT/'home/.cache/flashinfer'),(Path('/home/operator/.triton/cache'),ROOT/'triton')]:
 if source.exists():
  shutil.copytree(source,dest,dirs_exist_ok=True,symlinks=False)
venv=Path('/home/operator/.venvs/qwen38');cuda=venv/'lib/python3.12/site-packages/nvidia/cu13'
env={'PATH':str(venv/'bin')+':'+str(cuda/'bin')+':/usr/bin:/bin:/usr/lib/wsl/lib','VIRTUAL_ENV':str(venv),'HOME':str(ROOT/'home'),'TMPDIR':str(ROOT/'tmp'),'XDG_CACHE_HOME':str(ROOT/'home/.cache'),'HF_HOME':str(ROOT/'home/huggingface'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','SGLANG_CACHE_DIR':str(ROOT/'home/.cache/sglang'),'TRITON_CACHE_DIR':str(ROOT/'triton'),'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'torchinductor'),'TORCH_EXTENSIONS_DIR':str(ROOT/'torch-extensions'),'CUDA_CACHE_PATH':str(ROOT/'cuda-cache'),'FLASHINFER_WORKSPACE_BASE':str(ROOT/'home'),'CUDA_HOME':str(cuda),'CUDA_VISIBLE_DEVICES':'0','LD_LIBRARY_PATH':str(cuda/'lib')+':'+str(cuda/'lib64')+':/usr/lib/wsl/lib','LIBRARY_PATH':'/usr/lib/wsl/lib','MAX_JOBS':'2','OMP_NUM_THREADS':'4','NO_PROXY':'127.0.0.1,localhost','TOKENIZERS_PARALLELISM':'false'}
args=m['original_argv'][1:]
for flag,value in [('--port','34731'),('--served-model-name','qwen-normal-exp006')]:args[args.index(flag)+1]=value
args+=['--enable-metrics','--file-storage-path',str(ROOT/'storage'),'--crash-dump-folder',str(ROOT/'crash'),'--download-dir',str(ROOT/'downloads'),'--load-format','safetensors']
(ROOT/'launch-manifest.json').write_text(json.dumps({'argv':[str(venv/'bin/python')]+args,'environment':env,'cwd':str(ROOT)},indent=2))
with (ROOT/'server.log').open('xb') as log:
 p=subprocess.Popen([str(venv/'bin/python')]+args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'server-pid.json').write_text(json.dumps({'pid':p.pid,'stat':Path('/proc',str(p.pid),'stat').read_text()}))
print(json.dumps({'pid':p.pid,'port':34731,'model':'qwen-normal-exp006'}))
