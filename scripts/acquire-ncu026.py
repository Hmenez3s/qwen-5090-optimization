from pathlib import Path
import urllib.request,json,hashlib,shutil,time,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/ncu-tool026')
assert not r.exists();assert shutil.disk_usage(r.parent).free>=5*1024**3
r.mkdir();start=time.monotonic();url='https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-compute-2026.2.1_2026.2.1.5-1_amd64.deb';size=345070298;expected='6829651ceeb0c3f65890b9f727b74d1e550fed58c454e11c2c87442295e4eb70';count=0;h=hashlib.sha256()
try:
 with urllib.request.urlopen(url,timeout=30) as response,(r/'nsight-compute.deb').open('xb') as out:
  assert response.url.startswith('https://developer.download.nvidia.com/')
  while chunk:=response.read(1024*1024):
   count+=len(chunk);assert count<=size
   out.write(chunk);h.update(chunk)
   if time.monotonic()-start>360:raise TimeoutError('download exceeded360s')
 assert count==size and h.hexdigest()==expected
 p=subprocess.run(['dpkg-deb','--contents',str(r/'nsight-compute.deb')],capture_output=True,text=True,timeout=30);p.check_returncode();(r/'package-contents.txt').write_text(p.stdout)
 status={'state':'VERIFIED','bytes':count,'sha256':h.hexdigest(),'elapsed':time.monotonic()-start,'url':url}
except Exception as e:
 status={'state':'FAILED','bytes':count,'error':str(e),'elapsed':time.monotonic()-start}
(r/'acquisition.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
assert status['state']=='VERIFIED'
print('CONTENTS_HEAD','\n'.join(p.stdout.splitlines()[:18]));print('LINKS','\n'.join(x for x in p.stdout.splitlines() if x.startswith('l'))[:6000])
