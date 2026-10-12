"""Verify and stop one pinned final-stage service without cleanup."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,time,urllib.request
ALLOWED={'pi-serving-053':('qwen3.8-27b-normal',8081),'exp054-final-validation':('qwen-normal-exp054',34731)}
p=argparse.ArgumentParser();p.add_argument('--root',choices=sorted(ALLOWED),required=True);p.add_argument('--model',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--require-full',action='store_true');a=p.parse_args();assert ALLOWED[a.root]==(a.model,a.port)
r=Path('/home/operator/qwen38-5090/optimization-runs')/a.root
quality=[json.loads(x) for x in (r/'results/quality-final.jsonl').read_text().splitlines()];assert quality[-1].get('record_type')=='summary' and quality[-1].get('passed') is True;assert any(x.get('record_type')=='positive_marker_proof' and x.get('marker',{}).get('observer_removed') is True for x in quality)
if a.require_full:
 perf=[json.loads(x) for x in (r/'results/performance.jsonl').read_text().splitlines()];pc=[x for x in perf if x.get('record_type')=='case'];assert len(pc)==27 and all(x.get('status')=='ok' and x.get('completion_tokens')==1024 for x in pc)
 needles=[json.loads(x) for x in (r/'results/quality-needles.jsonl').read_text().splitlines()];nc=[x for x in needles if x.get('record_type')=='case'];want={'ALPHA':'R17M9','BETA':'V82K4','GAMMA':'N53T7'};assert len(nc)==3 and all(x.get('status')=='ok' and json.loads(x['text'])==want for x in nc)
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));metrics=op.open(f'http://127.0.0.1:{a.port}/metrics',timeout=10).read(2097152).decode();counts=[x for x in metrics.splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert len(counts)==2 and all(float(x.rsplit(' ',1)[1])==0 for x in counts)
o=json.loads((r/'server-pid.json').read_text());pid=o['pid'];proc=Path('/proc')/str(pid);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (proc/'stat').read_text().split()[21]==o['stat'].split()[21] and a.model.encode() in (proc/'cmdline').read_bytes()
before=json.loads((r/'protected-cache-before.json').read_text());after={}
for key,path in before['paths'].items():
 root=Path(path);records={}
 for parent,ds,fs in os.walk(root,followlinks=False):
  for name in ds+fs:assert not (Path(parent)/name).is_symlink()
  for name in fs:
   f=Path(parent)/name;records[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
 after[key]=records
assert after==before['hashes']
for path,want_hash in {'/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/speculative/dflash_utils.py':'fb72a3a28d5ecb33d77c6cd6655cc5b4abc8086feba5fb879c44b98b5cbb8374','/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py':'b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5'}.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==want_hash
os.kill(pid,signal.SIGTERM);start=time.monotonic()
while proc.exists():
 try:alive=(proc/'stat').read_text().split()[21]==o['stat'].split()[21]
 except FileNotFoundError:alive=False
 if not alive:break
 assert time.monotonic()-start<180;time.sleep(1)
free,power=map(float,subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,power.limit','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>=27000 and power==460
print(json.dumps({'status':'PASS','root':a.root,'quality_records':len(quality),'full':a.require_full,'stopped_pid':pid,'protected_unchanged':True,'gpu_free_MiB':free,'power_limit_W':power}))
