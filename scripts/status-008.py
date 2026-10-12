from pathlib import Path
import json,urllib.request,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/exp008-dflash8')
p=r/'server-pid.json'
print('PID',p.read_text() if p.exists() else 'not yet exec')
try:
 d=json.load(urllib.request.urlopen('http://127.0.0.1:34731/server_info',timeout=5))
 print('INFO',json.dumps({k:d.get(k) for k in ['status','served_model_name','context_length','max_total_num_tokens','startup_time','speculative_algorithm','speculative_dflash_block_size']}))
except Exception as e:print('API',str(e))
log=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/experiment-008-dflash8/stderr.log')
print('TAIL','\n'.join(log.read_text().splitlines()[-6:]) if log.exists() else 'missing')
print(subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,temperature.gpu','--format=csv'],capture_output=True,text=True).stdout)
