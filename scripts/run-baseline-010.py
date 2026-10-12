from pathlib import Path
import json,urllib.request,subprocess,os,time,datetime
root=Path('/home/operator/qwen38-5090/optimization-runs/exp010-context')
m=json.loads((root/'launch.json').read_text());owner=json.loads((root/'server-pid.json').read_text())
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
assert Path('/proc',str(owner['pid']),'stat').read_text().split()[21]==owner['stat'].split()[21]
info=json.load(urllib.request.urlopen('http://127.0.0.1:34731/server_info',timeout=10))
assert info['served_model_name']=='qwen-normal-exp010' and info['status']=='ready'
assert info['max_total_num_tokens']>=131072 and info['context_length']>=131072
for key in ['delete_ckpt_after_loading','enable_hierarchical_cache','enable_lmcache','enable_flexkv','quantize_and_serve','trust_remote_code']:assert not info[key],key
assert not info['tool_server'] and not info['forward_hooks']
for key in ['target_verify','draft_decode']:assert info['startup_time']['cuda_graph'][key]>0,key
results=root/'results';results.mkdir()
(results/'preflight.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner':owner,'version':info['version'],'model_path':info['model_path'],'context':info['context_length'],'graphs':info['startup_time']['cuda_graph'],'port':34731},indent=2))
(root/'cases/warmup.json').write_text(json.dumps({'cases':[{'name':'warmup','payload':{'text':'Reply with READY.','sampling_params':{'temperature':0,'max_new_tokens':32}}}],'repetitions':1}))
env={'PATH':'/usr/bin:/bin','HOME':m['environment']['HOME'],'TMPDIR':m['environment']['TMPDIR'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
started=time.monotonic()
for name in ['warmup','quality-needles','performance']:
 command=['/usr/bin/python3',str(root/'benchmark_stream.py'),'--url','http://127.0.0.1:34731','--expected-model','qwen-normal-exp010','--manifest',str(root/'cases'/f'{name}.json'),'--output',str(results/f'{name}.jsonl')]
 r=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=max(1,900-(time.monotonic()-started)))
 print('PHASE',name,'EXIT',r.returncode,r.stderr[-1000:],flush=True)
 for line in (results/f'{name}.jsonl').read_text().splitlines():
  d=json.loads(line)
  if d.get('record_type')=='case':
   print(json.dumps({k:d.get(k) for k in ['case','repetition','status','completion_tokens','metrics','finish_reason','error']}),flush=True)
 r.check_returncode()
 if name=='quality-needles':
  expected={'ALPHA':'R17M9','BETA':'V82K4','GAMMA':'N53T7'}
  for line in (results/f'{name}.jsonl').read_text().splitlines():
   d=json.loads(line)
   if d.get('record_type')!='case':continue
   text=(d.get('text') or '').rsplit('</think>',1)[-1]
   found=[]
   for i,ch in enumerate(text):
    if ch=='{':
     try:found.append(json.JSONDecoder().raw_decode(text[i:])[0])
     except json.JSONDecodeError:pass
   passed=expected in found and d.get('finish_reason_type')!='length'
   print('NEEDLE_ORACLE',d['case'],passed,flush=True)
   if not passed:raise RuntimeError('Long-context quality baseline failed; stop before performance suite')
print('FINISHED_SECONDS',time.monotonic()-started,flush=True)
