from pathlib import Path
import json,struct,urllib.request
p=Path('/home/operator/qwen38-5090/models/NInfer-Normal-11dbbbbb/qwen3_8_27b_nvfp4.ninfer')
with p.open('rb') as f:
 h=f.read(16);assert h[:8]==b'NINFER\x00\x02';n=struct.unpack('<Q',h[8:])[0];assert n<16*1024**2;d=json.loads(f.read(n))
print('ARTIFACT',json.dumps(d['identity']));draft=[x['name'] for x in d['objects'] if x['name'].startswith('dflash2/')];print('DFLASH_OBJECTS',len(draft),draft[:12]);print('MTP_OBJECTS',sum(x['name'].startswith('mtp/') for x in d['objects']))
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
 with op.open('http://127.0.0.1:34731/server_info',timeout=10) as f:a=json.load(f)
 print('SGLANG',a.get('status'),a.get('served_model_name'))
except Exception as e:print('STATUS',repr(e))
