from pathlib import Path
import json,urllib.request,hashlib
r=Path('/home/operator/qwen38-5090/optimization-runs');op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with op.open('http://127.0.0.1:34731/server_info',timeout=10) as f:d=json.load(f)
print('SERVICE',d.get('status'),d.get('served_model_name'))
p=r/'ninfer-build-014/protected-dpkg-sha256.txt';print('DPKG_UNCHANGED',hashlib.sha256(Path('/var/lib/dpkg/status').read_bytes()).hexdigest()==p.read_text())
for p in [r/'exp019-profile8/results/profile-code-2048.jsonl',r/'exp019-profile8/results/profile-prose-130048.jsonl']:
 for l in p.read_text().splitlines():
  x=json.loads(l)
  if x.get('record_type')=='case':print('PROFILE_REQUEST',x['case'],x['status'],x['completion_tokens'],'cached',x['meta']['cached_tokens'])
