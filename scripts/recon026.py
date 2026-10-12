from pathlib import Path
import json,shutil,subprocess,urllib.request
b=Path('/home/operator/qwen38-5090/optimization-runs');r=b/'pi-serving-025';m=json.loads((r/'launch.json').read_text());o=json.loads((r/'server-pid.json').read_text());p=Path('/proc')/str(o['pid'])
d={'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid':o['pid'],'same_identity':False,'tools':{n:shutil.which(n) for n in ['ncu','nsys','nvcc','perf']}}
d['same_identity']=d['boot']==o['boot_id'] and p.exists() and (p/'stat').read_text().split()[21]==o['stat'].split()[21]
for root in ['/usr/local','/opt/nvidia','/opt/nvidia/nsight-compute','/opt/nvidia/nsight-systems','/home/operator/.venvs/qwen38/lib/python3.12/site-packages/nvidia']:
 q=Path(root)
 if q.is_dir():d[root]=[x.name for x in q.iterdir()][:60]
d['gpu']=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.free,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu','--format=csv,noheader'],text=True)
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with op.open('http://127.0.0.1:8081/server_info',timeout=10) as f:info=json.load(f)
d['info_keys']=list(info);d['status']=info.get('status');d['model']=info.get('served_model_name');d['context_length']=info.get('context_length');d['max_total_tokens']=info.get('max_total_tokens')
d['environment_paths']={k:v for k,v in m['environment'].items() if k in ['CUDA_HOME','FLASHINFER_WORKSPACE_BASE','FLASHINFER_AUTOTUNE_CACHE','CUTE_DSL_CACHE_DIR','CUTE_DSL_DUMP_DIR','TMPDIR']}
print(json.dumps(d))
