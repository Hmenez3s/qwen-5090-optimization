from pathlib import Path
import json,copy,sys,urllib.request
r=Path('/home/operator/qwen38-5090/optimization-runs/exp019-profile8');sys.path.insert(0,str(r));import benchmark_stream as b
m=json.loads((r/'launch.json').read_text());assert 'SGLANG_PROFILE_V2' not in m['environment']
o=json.loads((r/'server-pid.json').read_text());assert (Path('/proc')/str(o['pid'])/'stat').read_text().split()[21]==o['stat'].split()[21];assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id']
source=json.loads((r.parent/'exp010-context/cases/performance.json').read_text());op=b.build_isolated_opener();url='http://127.0.0.1:34731';model='qwen-normal-exp019'
for name in ['code-2048','prose-130048']:
 case=next(x for x in source['cases'] if x['name']==name);warm=copy.deepcopy(case);warm['payload']['sampling_params']['max_new_tokens']=64
 b.run_benchmark(url=url,expected_model=model,manifest={'cases':[warm],'repetitions':1},output=r/'results'/('profile-warm-'+name+'.jsonl'))
 body={'output_dir':str(r/'profiles'),'profile_id':'exp019-'+name,'num_steps':32,'activities':['CPU','GPU'],'with_stack':False,'record_shapes':False,'profile_by_stage':False,'merge_profiles':False}
 with op.open(urllib.request.Request(url+'/start_profile',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=60) as response:print('START',name,response.status,response.read().decode(),flush=True)
 b.run_benchmark(url=url,expected_model=model,manifest={'cases':[case],'repetitions':1},output=r/'results'/('profile-'+name+'.jsonl'))
 traces=list((r/'profiles').glob('exp019-'+name+'*'));assert traces,'trace not exported';print('EXPORTED',[(x.name,x.stat().st_size) for x in traces],flush=True)
