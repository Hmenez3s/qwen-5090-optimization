from pathlib import Path
import ast,json,urllib.request
s=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt')
result={}
for root,symbol in [(s/'utils','kill_process_tree'),(s/'speculative','is_dflash_sampling_verify_available')]:
 for p in root.glob('*.py'):
  text=p.read_text()
  if 'def '+symbol+'(' not in text:continue
  tree=ast.parse(text)
  for n in ast.walk(tree):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==symbol:
    result[str(p)+':'+str(n.lineno)]='\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])
result['loader_files']=[p.name for p in (s/'model_loader').glob('*.py')]
p=s/'model_loader/loader.py'
if p.exists():
 lines=p.read_text().splitlines();out=[]
 for i,line in enumerate(lines):
  if any(x in line for x in ['local_files_only','os.path.isdir','is_local','HF_HUB_OFFLINE','snapshot_download','download_weights','is_offline_mode']):
   out.append({'line':i+1,'context':'\n'.join(lines[max(0,i-5):i+12])})
 result['loader_snippets']=out
info=json.load(urllib.request.urlopen('http://127.0.0.1:8081/server_info',timeout=10))
keys=['delete_ckpt_after_loading','model_loader_extra_config','trust_remote_code','custom_weight_loader','modelopt_export_path','modelopt_checkpoint_save_path','quantize_and_serve','weight_loader_drop_cache_after_load','load_format','download_dir','enable_lora','enable_prefix_mm_cache','kv_events_config','hicache_storage_backend','remote_instance_weight_loader_seed_instance_ip','file_storage_path','crash_dump_folder']
result['safe_flags']={k:info.get(k) for k in keys}
print(json.dumps(result,indent=2))
