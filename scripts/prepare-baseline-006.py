from pathlib import Path
import json,os,hashlib,socket,datetime
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908/sglang-baseline')
assert not ROOT.exists()
p=Path('/proc/834');cmd=p.joinpath('cmdline').read_bytes().decode().split('\0');cmd=[x for x in cmd if x]
assert 'sglang.launch_server' in cmd
assert cmd[cmd.index('--model-path')+1]=='/home/operator/qwen38-5090/models/Qwen3.8-27B-NVFP4-LMHead4'
assert cmd[cmd.index('--port')+1]=='8081'
with socket.socket() as sock:sock.bind(('127.0.0.1',34731))
base=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows')
protected={}
for model in [Path(cmd[cmd.index('--model-path')+1]),Path(cmd[cmd.index('--speculative-draft-model-path')+1])]:
 for f in model.iterdir():
  if f.is_file():
   st=f.stat();protected[str(f)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino}
   if st.st_size<200000:protected[str(f)]['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
for name in ['start-shared-131k.ps1','start-shared-lmhead4-131k.ps1','start-server.sh']:
 f=base/name;protected[str(f)]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_text()}
env_raw=dict(x.split('=',1) for x in p.joinpath('environ').read_bytes().decode().split('\0') if '=' in x)
allowed=['CUDA_HOME','LD_LIBRARY_PATH','LIBRARY_PATH','MAX_JOBS','SGLANG_DISABLE_DFLASH_FUSED_KV','SGLANG_MAX_THINK_TOKENS','NCCL_BUFFSIZE']
selected={k:env_raw[k] for k in allowed if k in env_raw}
extras=[k for k in env_raw if k.startswith(('SGLANG_','FLASHINFER_','NCCL_')) and k not in selected]
assert not extras, 'unresolved runtime environment keys '+str(extras)
proc=[]
for a in Path('/proc').iterdir():
 if not a.name.isdigit():continue
 try:
  fields=(a/'stat').read_text().rsplit(')',1)[1].split();proc.append({'pid':int(a.name),'ppid':int(fields[1]),'name':(a/'comm').read_text().strip()})
 except OSError:pass
children={834}
while True:
 expanded=children|{x['pid'] for x in proc if x['ppid'] in children}
 if expanded==children:break
 children=expanded
ROOT.mkdir()
manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_pid':834,'original_stat':p.joinpath('stat').read_text(),'original_argv':cmd,'original_environment':selected,'children':[x for x in proc if x['pid'] in children],'protected':protected,'port':34731,'root':str(ROOT)}
(ROOT/'recovery-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
