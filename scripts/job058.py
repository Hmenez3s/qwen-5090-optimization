"""Run only the isolated synthetic MICRO058 child and preserve its exit receipt."""
from pathlib import Path
import subprocess,json,time
r=Path('/home/operator/qwen38-5090/optimization-runs/micro058')
start=time.monotonic()
with (r/'micro-stdout.log').open('x') as out,(r/'micro-stderr.log').open('x') as err:
 try:
  p=subprocess.run(['/home/operator/.venvs/qwen38/bin/python',str(r/'micro_gemm058.py'),'--output',str(r/'results.jsonl'),'--server-cache',str(r/'server-autotune.json')],cwd=r,stdout=out,stderr=err,timeout=1800)
  status={'stage':'COMPLETE','micro_exit':p.returncode}
 except subprocess.TimeoutExpired:
  status={'stage':'TIMEOUT','micro_exit':None}
status['elapsed']=time.monotonic()-start
with (r/'job-status.json').open('x') as f:json.dump(status,f)
print(json.dumps(status),flush=True)
