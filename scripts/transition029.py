from pathlib import Path
import json,os,signal
r=Path('/home/operator/qwen38-5090/optimization-runs/micro029');m=json.loads((r/'launch.json').read_text());old=m['wait_for'];p=Path('/proc')/str(old['pid'])
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert (p/'stat').read_text().split()[21]==old['start_ticks']
assert b'qwen3.8-27b-normal' in (p/'cmdline').read_bytes()
controllers=[]
for x in Path('/proc').iterdir():
 if not x.name.isdigit():continue
 try:
  c=(x/'cmdline').read_bytes()
  if str(r/'exec-manifest.py').encode() in c and str(r/'launch.json').encode() in c:controllers.append(x.name)
 except (OSError,PermissionError):pass
assert len(controllers)==1,controllers
print('CONTROLLER_VERIFIED',controllers,'OLD',old['pid'],flush=True)
os.kill(old['pid'],signal.SIGTERM)
print('SIGTERM_SENT')
