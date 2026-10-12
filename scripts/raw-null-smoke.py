from pathlib import Path
import sys,json,urllib.request
r=Path('/home/operator/qwen38-5090/optimization-runs/client-smoke-015');sys.path.insert(0,str(r));import benchmark_stream as b
op=b.build_isolated_opener();b._models(op,'http://127.0.0.1:34731','qwen-normal-exp012',10)
p=json.loads((r/'cases.json').read_text())['cases'][0]['payload'];p.update(model='qwen-normal-exp012',max_tokens=32,stream=True,stream_options={'include_usage':True})
with op.open(urllib.request.Request('http://127.0.0.1:34731/v1/chat/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'}),timeout=60) as response: raw=response.read()
with (r/'raw-null-smoke.sse').open('xb') as f:f.write(raw)
print(raw[:1500].decode())
