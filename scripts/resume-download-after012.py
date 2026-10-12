from pathlib import Path
import json,os
r=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-resume-013')
d=json.loads((r/'status.json').read_text());assert d['state']=='PAUSED'
p=r/'control-resume-after012.next'
with p.open('x') as f:json.dump({'pause':False},f)
os.replace(p,r/'control.json');print('DOWNLOAD_RESUME_REQUESTED')
