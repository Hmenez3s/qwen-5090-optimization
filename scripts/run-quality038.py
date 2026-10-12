from pathlib import Path
import subprocess,os
r=Path('/home/operator/qwen38-5090/optimization-runs/exp038-minp')
# Pure-stdlib harness: -S prevents inference sitecustomize in this observer process.
env={'PATH':'/usr/bin:/bin','HOME':str(r/'home'),'TMPDIR':'/tmp/qwen-exp038-minp/tmp','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
p=subprocess.run(['/usr/bin/python3','-S',str(r/'quality-minp038.py')],cwd=r,env=env,timeout=2100,capture_output=True,text=True)
with (r/'results/quality-observer.log').open('x') as f:f.write(p.stdout+p.stderr)
print(p.stdout,p.stderr,flush=True)
raise SystemExit(p.returncode)
