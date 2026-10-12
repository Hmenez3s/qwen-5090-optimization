import time,json,urllib.request,subprocess
from pathlib import Path
started=time.monotonic()
while time.monotonic()-started<900:
 receipt=Path('/home/operator/qwen38-5090/optimization-runs/exp023-kda8/server-pid.json')
 if receipt.exists():
  pid=json.loads(receipt.read_text())['pid']
  if not (Path('/proc')/str(pid)).exists():print('SERVER_EXITED');break
 try:
  d=json.load(urllib.request.urlopen('http://127.0.0.1:34731/server_info',timeout=3))
  if d.get('status')=='ready' and d.get('served_model_name')=='qwen-normal-exp023':
   print(json.dumps({'status':'READY','model':d.get('served_model_name'),'graphs':d.get('startup_time'),'utc':time.time()}));break
 except Exception:pass
 time.sleep(10)
else:
 print('NOT_READY_AFTER_900S')
 print(subprocess.run(['ps','-eo','pid,ppid,etimes,pcpu,rss,comm','--sort=-pcpu'],capture_output=True,text=True).stdout[:2200])
log=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/experiment-023-kda8/stderr.log')
print('\n'.join(log.read_text().splitlines()[-6:]))
