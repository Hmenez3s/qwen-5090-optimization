"""Run final synthetic integration, context, performance, reasoning and tool gates."""
from pathlib import Path
import json,subprocess,time
r=Path('/home/operator/qwen38-5090/optimization-runs/exp054-final-validation');model='qwen-normal-exp054'
o=json.loads((r/'server-pid.json').read_text());proc=Path('/proc')/str(o['pid']);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (proc/'stat').read_text().split()[21]==o['stat'].split()[21]
env={'PATH':'/usr/bin:/bin','HOME':str(r/'home'),'TMPDIR':'/tmp/qwen-exp054-final-validation/tmp','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
status={'started_monotonic':time.monotonic(),'steps':[]}
def call(name,cmd,timeout):
 q=subprocess.run(cmd,cwd=r,env=env,capture_output=True,text=True,timeout=timeout);status['steps'].append({'name':name,'exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr});print(name,q.returncode,q.stdout,q.stderr,flush=True);q.check_returncode()
call('quality-final',['/usr/bin/python3','-S',str(r/'quality-final054.py')],1000)
quality=[json.loads(x) for x in (r/'results/quality-final.jsonl').read_text().splitlines()];assert quality[-1].get('record_type')=='summary' and quality[-1].get('passed') is True;assert any(x.get('record_type')=='positive_marker_proof' and x.get('marker',{}).get('observer_removed') is True for x in quality)
for suite,expected in [('quality-needles',3),('performance',27)]:
 out=r/'results'/(suite+'.jsonl');cmd=['/usr/bin/python3',str(r/'benchmark_stream.py'),'--protocol','openai','--url','http://127.0.0.1:34731','--expected-model',model,'--manifest',str(r/'cases'/(suite+'.json')),'--output',str(out)];call(suite,cmd,1200);rows=[json.loads(x) for x in out.read_text().splitlines()];cases=[x for x in rows if x.get('record_type')=='case'];assert len(cases)==expected and all(x.get('status')=='ok' for x in cases)
 if suite=='quality-needles':
  expected_text={'ALPHA':'R17M9','BETA':'V82K4','GAMMA':'N53T7'}
  for case in cases:assert json.loads(case['text'])==expected_text,case['case']
status['status']='PASS';status['elapsed']=time.monotonic()-status['started_monotonic'];(r/'results/run-final-status.json').write_text(json.dumps(status,indent=2));print(json.dumps({'status':'PASS','quality_records':len(quality),'needles':3,'performance':27,'elapsed':status['elapsed']}),flush=True)
