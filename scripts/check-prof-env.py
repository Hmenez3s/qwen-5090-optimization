from pathlib import Path
import json
r=Path('/home/operator/qwen38-5090/optimization-runs/exp019-profile8');m=json.loads((r/'launch.json').read_text());print('OVERRIDE',m['environment'].get('SGLANG_PROFILE_V2','UNSET'))
p=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/environ.py')
for i,l in enumerate(p.read_text().splitlines(),1):
 if 'SGLANG_PROFILE_V2' in l:print('DEFAULT',i,l.strip())
