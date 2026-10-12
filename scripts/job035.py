"""Run only the isolated synthetic MINP035 child and preserve its exit receipt."""
from pathlib import Path
import subprocess,json,time
r=Path('/home/operator/qwen38-5090/optimization-runs/minp035')
start=time.monotonic()
with (r/'micro-stdout.log').open('x') as out,(r/'micro-stderr.log').open('x') as err:
 try:
  p=subprocess.run(['/home/operator/.venvs/qwen38/bin/python',str(r/'test_dflash_minp035.py'),'--output',str(r/'results.jsonl')],cwd=r,stdout=out,stderr=err,timeout=240)
  status={'stage':'COMPLETE','micro_exit':p.returncode}
 except subprocess.TimeoutExpired:
  status={'stage':'TIMEOUT','micro_exit':None}
status['elapsed']=time.monotonic()-start
with (r/'job-status.json').open('x') as f:json.dump(status,f)
print(json.dumps(status),flush=True)
