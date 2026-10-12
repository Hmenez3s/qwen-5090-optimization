from pathlib import Path
import json, urllib.request, subprocess, hashlib, datetime
root=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows')
info=json.load(urllib.request.urlopen('http://127.0.0.1:8081/server_info',timeout=10))
keys=['model_path','tokenizer_path','version','context_length','max_total_tokens','max_total_num_tokens','max_req_input_len','speculative_algorithm','speculative_draft_model_path','speculative_num_draft_tokens','speculative_dflash_block_size','speculative_draft_window_size','speculative_use_rejection_sampling','speculative_accept_threshold_single','speculative_accept_threshold_acc','sampling_defaults','quantization','fp4_gemm_runner_backend','attention_backend','kv_cache_dtype','mem_fraction_static','max_running_requests','mamba_ssm_dtype','max_mamba_cache_size','enable_hierarchical_cache','enable_lmcache','enable_flexkv','tool_server','forward_hooks','chat_template','default_chat_template_kwargs','enable_strict_thinking','preferred_sampling_params','reasoning_parser','tool_call_parser','startup_time','status','cuda_graph_config']
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'server_info':{k:info.get(k) for k in keys},'files':{},'processes':[]}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().decode().split('\0')
  if 'sglang.launch_server' in cmd:
   safe=[]
   secret=False
   for v in cmd:
    safe.append('[REDACTED]' if secret else v)
    secret=v in ('--api-key','--admin-api-key','--ssl-keyfile-password')
   result['processes'].append({'pid':int(p.name),'cmd':safe,'stat':(p/'stat').read_text(),'cwd':str((p/'cwd').resolve()),'exe':str((p/'exe').resolve())})
 except (OSError,UnicodeError):pass
names=['start-shared-131k.ps1','start-shared-lmhead4-131k.ps1','start-server.sh','AUDITORIA-PERFORMANCE.md']
for name in names:
 p=root/name
 if p.exists() and p.stat().st_size<200000:
  b=p.read_bytes();result['files'][str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
for label in ['model_path','speculative_draft_model_path']:
 d=Path(info[label]);result[label+'_files']=[{'name':p.name,'size':p.stat().st_size,'symlink':p.is_symlink()} for p in d.iterdir() if p.is_file()]
 for name in ['README.md','config.json','hf_quant_config.json','generation_config.json']:
  p=d/name
  if p.exists() and p.stat().st_size<100000:
   b=p.read_bytes();result['files'][str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
result['gpu']=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.limit','--format=csv'],capture_output=True,text=True).stdout
result['disk']=subprocess.run(['df','-h','/','/mnt/c'],capture_output=True,text=True).stdout
result['roots']={str(r):[p.name for p in r.iterdir()] for r in [Path('/home/operator/qwen38-5090'),Path('/home/operator/qwen38-5090/models'),Path('/home/operator/.venvs')]}
print(json.dumps(result))
