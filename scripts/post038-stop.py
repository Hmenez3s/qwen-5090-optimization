"""Verify exp038 evidence/boundaries, then stop only its pinned private process."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,urllib.request
r=Path('/home/operator/qwen38-5090/optimization-runs/exp038-minp')
rows=[json.loads(line) for line in (r/'results/quality-minp.jsonl').read_text().splitlines()]
assert rows[-1].get('record_type')=='summary' and rows[-1].get('passed') is True
assert all(x.get('passed') is not False for x in rows),[x.get('name') for x in rows if x.get('passed') is False]
assert any(x.get('record_type')=='positive_marker_proof' and x.get('passed') for x in rows)
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
metrics=op.open('http://127.0.0.1:34731/metrics',timeout=10).read(2097152).decode()
counts=[line for line in metrics.splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
assert len(counts)==2 and all(float(line.rsplit(' ',1)[1])==0 for line in counts),counts
receipt=json.loads((r/'server-pid.json').read_text());pid=receipt['pid'];p=Path('/proc')/str(pid)
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==receipt['boot_id']
assert (p/'stat').read_text().split()[21]==receipt['stat'].split()[21]
assert b'qwen-normal-exp038' in (p/'cmdline').read_bytes()
before=json.loads((r/'protected-cache-before.json').read_text());after={}
for key,path in before['paths'].items():
 root=Path(path);records={}
 for parent,ds,fs in os.walk(root,followlinks=False):
  for name in ds+fs:assert not (Path(parent)/name).is_symlink()
  for name in fs:
   f=Path(parent)/name;records[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
 after[key]=records
assert after==before['hashes'],'protected cache changed'
protected={
 '/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/speculative/dflash_utils.py':'fb72a3a28d5ecb33d77c6cd6655cc5b4abc8086feba5fb879c44b98b5cbb8374',
 '/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py':'b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5'}
for path,expected in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
os.kill(pid,signal.SIGTERM)
start=time.monotonic()
while p.exists():
 try:alive=(p/'stat').read_text().split()[21]==receipt['stat'].split()[21]
 except FileNotFoundError:alive=False
 if not alive:break
 assert time.monotonic()-start<180
 time.sleep(1)
gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,power.limit','--format=csv,noheader,nounits'],text=True).strip();free,power=map(float,gpu.split(','));assert free>=27000 and power==460
print(json.dumps({'status':'PASS','quality_records':len(rows),'positive_marker_proof':True,'protected_cache_unchanged':True,'protected_sources_unchanged':True,'stopped_pid':pid,'gpu_free_MiB':free,'power_limit_W':power}))
