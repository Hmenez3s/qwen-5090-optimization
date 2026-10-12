from pathlib import Path
import json,subprocess,shutil
site=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages')
paths=['sglang/launch_server.py','sglang/srt/environ.py','flashinfer/jit/env.py','sglang/srt/speculative/dflash_worker_v2.py','sglang/srt/entrypoints/http_server.py','sglang/srt/managers/tokenizer_manager.py']
files={}
for name in paths:
 p=site/name
 if p.exists() and p.stat().st_size<200000: files[name]=p.read_text()
print(json.dumps({'files':files,'tools':{n:shutil.which(n) for n in ['bwrap','unshare','cmake','ninja','git','nvcc','gcc']},'site':str(site)}))
