"""Read-only postcheck for synthetic MICRO029 and protected cache boundaries."""
from pathlib import Path
import json,hashlib,os,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/micro029b')
receipt=json.loads((r/'job-status.json').read_text());assert receipt['stage']=='COMPLETE'
o=json.loads((r/'server-pid.json').read_text());p=Path('/proc')/str(o['pid'])
assert not (Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and p.exists() and (p/'stat').read_text().split()[21]==o['stat'].split()[21])
before=json.loads((r/'protected-cache-before.json').read_text());after={}
for key,path in before['paths'].items():
 root=Path(path);records={}
 for parent,ds,fs in os.walk(root,followlinks=False):
  for name in ds+fs:assert not (Path(parent)/name).is_symlink()
  for name in fs:
   f=Path(parent)/name;records[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
 after[key]=records
assert after==before['hashes'],'protected cache changed'
kda=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py')
h=hashlib.sha256(kda.read_bytes()).hexdigest();assert h=='b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5'
gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,power.limit','--format=csv,noheader,nounits'],text=True).strip();free,power=map(float,gpu.split(','));assert free>=27000 and power==460
print(json.dumps({'status':'PASS','job':receipt,'protected_cache_unchanged':True,'kda_hash':h,'gpu_free_MiB':free,'power_limit_W':power,'result_jsonl':(r/'results.jsonl').read_text()}))
