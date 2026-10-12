from pathlib import Path
import json
base=Path('/home/operator/qwen38-5090/optimization-runs');expected={'ALPHA':'R17M9','BETA':'V82K4','GAMMA':'N53T7'}
for name in ['exp017-ninfer4','exp018-mtp3']:
 rows=[json.loads(l) for l in (base/name/'results/quality-needles.jsonl').read_text().splitlines()];rows=[x for x in rows if x.get('record_type')=='case']
 for x in rows:assert json.loads(x['text'])==expected and x['finish_reason']=='stop'
 perf=[json.loads(l) for l in (base/name/'results/performance.jsonl').read_text().splitlines()];perf=[x for x in perf if x.get('record_type')=='case'];assert len(perf)==27 and all(x['status']=='ok' and x['completion_tokens']==1024 for x in perf)
 print(name,'NEEDLES_PASS',len(rows),'PERF_PASS',len(perf),'LONG_TOTAL_MIN',min(x['usage']['total_tokens'] for x in perf if '130048' in x['case']))
