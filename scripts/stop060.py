from pathlib import Path
import json,os,time,urllib.request
root=Path('/home/operator/qwen38-5090/optimization-runs/exp060-dspark')
receipt=json.loads((root/'server-pid.json').read_text());pid=int(receipt['pid']);proc=Path('/proc')/str(pid)
same=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==receipt['boot_id'] and proc.exists() and (proc/'stat').read_text().split()[21]==receipt['stat'].split()[21]
if not same:
 print(json.dumps({'status':'ALREADY_STOPPED','pid':pid}));raise SystemExit(0)
cmd=(proc/'cmdline').read_bytes().split(b'\0');assert b'qwen-normal-exp060' in cmd and str(root).encode()==(proc/'cwd').resolve().as_posix().encode()
try:
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));metrics=opener.open('http://127.0.0.1:34731/metrics',timeout=10).read().decode();values={}
 for line in metrics.splitlines():
  for name in ('sglang:num_running_reqs','sglang:num_queue_reqs'):
   if line.startswith(name+'{'):values[name]=float(line.rsplit(' ',1)[1])
 assert values=={'sglang:num_running_reqs':0.0,'sglang:num_queue_reqs':0.0},values
except (OSError,ValueError):
 # A failed/not-ready server has no request surface; PID identity remains exact.
 values='API_NOT_READY'
os.kill(pid,15)
for _ in range(480):
 try: alive=proc.exists() and (proc/'stat').read_text().split()[21]==receipt['stat'].split()[21]
 except FileNotFoundError:alive=False
 if not alive:print(json.dumps({'status':'STOPPED','pid':pid,'signal':'SIGTERM','metrics':values}));break
 time.sleep(.25)
else:raise TimeoutError('exp060 did not stop; no escalation attempted')
