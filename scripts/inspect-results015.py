from pathlib import Path
import json
base=Path('/home/operator/qwen38-5090/optimization-runs')
for name in ['exp016-common8','exp015b-ninfer8']:
 print('NEEDLES',name)
 for line in (base/name/'results/quality-needles.jsonl').read_text().splitlines():
  d=json.loads(line)
  if d.get('record_type')=='case':print(d['case'],d.get('text'),d.get('finish_reason'))
p=base/'exp015b-ninfer8/requests.jsonl'
with p.open() as f:
 for i,l in enumerate(f):
  if i<2:print('REQUEST_LOG',l[:10000])
  else:break
