from pathlib import Path
import subprocess,json,time
r=Path('/home/operator/qwen38-5090/optimization-runs/micro027');python='/home/operator/.venvs/qwen38/bin/python';ncu='/home/operator/qwen38-5090/optimization-runs/ncu-tool026/root/opt/nvidia/nsight-compute/2026.2.1/ncu'
start=time.monotonic()
with (r/'counter-stdout.log').open('x') as out,(r/'counter-stderr.log').open('x') as err:
 p=subprocess.run([ncu,'--config-file','0','--clock-control','none','--cache-control','none','--target-processes','application-only','--apply-rules','no','--metrics','gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed','--launch-count','1','--export',str(r/'counter-report'),python,str(r/'counter-probe027.py')],cwd=r,stdout=out,stderr=err,timeout=180)
(r/'counter-status.json').write_text(json.dumps({'exit_code':p.returncode,'elapsed':time.monotonic()-start}));print('COUNTER_EXIT',p.returncode,flush=True)
text=(r/'counter-stdout.log').read_text()+(r/'counter-stderr.log').read_text()
if p.returncode and 'ERR_NVGPUCTRPERM' not in text:
 raise RuntimeError('unclassified counter failure; preserve evidence without retry')
with (r/'micro-stdout.log').open('x') as out,(r/'micro-stderr.log').open('x') as err:
 p=subprocess.run([python,str(r/'micro_gemm027.py'),'--output',str(r/'results.jsonl'),'--server-cache',str(r/'server-autotune.json')],cwd=r,stdout=out,stderr=err,timeout=1200)
(r/'job-status.json').write_text(json.dumps({'stage':'COMPLETE','micro_exit':p.returncode,'elapsed':time.monotonic()-start}));print('MICRO_EXIT',p.returncode,flush=True)
