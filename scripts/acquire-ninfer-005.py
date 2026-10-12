import concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908')
manifest=json.loads((ROOT/'download-manifest.json').read_text())
dest=Path(manifest['destination'])
started=time.monotonic()
def fetch(item):
    name=item['rfilename']
    if '/' in name or name.startswith('.'):
        return {'file':name,'status':'SKIP_NON_RUNTIME'}
    p=dest/name
    url=manifest['base_url']+'/'+name
    size=0; digest=hashlib.sha256()
    with p.open('xb') as out:
        with urllib.request.urlopen(url,timeout=120) as res:
            while True:
                if time.monotonic()-started>10800:raise TimeoutError('total budget exceeded')
                chunk=res.read(4*1024*1024)
                if not chunk:break
                out.write(chunk);size+=len(chunk);digest.update(chunk)
    expected=item.get('lfs',{}).get('sha256')
    if size != item['size']:raise ValueError('size mismatch '+name)
    if expected and digest.hexdigest()!=expected:raise ValueError('hash mismatch '+name)
    row={'file':name,'status':'VERIFIED','bytes':size,'sha256':digest.hexdigest(),'expected_sha256':expected}
    print(json.dumps(row),flush=True)
    return row
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(fetch,manifest['files']))
    result={'status':'COMPLETE','files':rows,'seconds':time.monotonic()-started}
except Exception as exc:
    result={'status':'FAILED','error':repr(exc),'seconds':time.monotonic()-started}
(ROOT/'download-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':result['status'],'seconds':result['seconds']}),flush=True)
