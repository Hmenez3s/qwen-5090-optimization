from pathlib import Path
import json,urllib.request
r=Path('/home/operator/qwen38-5090/optimization-runs/exp008-dflash8/results')
last=json.loads((r/'performance.jsonl').read_text().splitlines()[-1])
print(json.dumps({'meta':last.get('meta'),'error':last.get('error'),'events':last.get('events',[])[:2]},indent=2))
try:
 d=json.load(urllib.request.urlopen('http://127.0.0.1:34731/server_info',timeout=5));print('SERVER',d.get('status'),d.get('max_req_input_len'))
except Exception as e:print('API',str(e))
