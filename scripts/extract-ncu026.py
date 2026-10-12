from pathlib import Path,PurePosixPath
import tarfile,subprocess,json,os,hashlib
r=Path('/home/operator/qwen38-5090/optimization-runs/ncu-tool026');out=r/'root';assert not out.exists();out.mkdir()
assert json.loads((r/'acquisition.json').read_text())['state']=='VERIFIED'
before=hashlib.sha256(Path('/var/lib/dpkg/status').read_bytes()).hexdigest();count=0;size=0
p=subprocess.Popen(['dpkg-deb','--fsys-tarfile',str(r/'nsight-compute.deb')],stdout=subprocess.PIPE)
with tarfile.open(fileobj=p.stdout,mode='r|') as tar:
 for member in tar:
  parts=PurePosixPath(member.name).parts
  assert not PurePosixPath(member.name).is_absolute() and '..' not in parts
  if parts[:4]!=('opt','nvidia','nsight-compute','2026.2.1'):continue
  assert member.isfile() or member.isdir() or member.issym()
  if member.issym():assert not PurePosixPath(member.linkname).is_absolute()
  size+=member.size;count+=1;assert size<2*1024**3 and count<20000
  tar.extract(member,path=out,filter='data')
assert p.wait(timeout=10)==0
for name in ['home','tmp']:(r/name).mkdir()
env={'PATH':'/usr/bin:/bin','HOME':str(r/'home'),'XDG_CACHE_HOME':str(r/'home/.cache'),'XDG_CONFIG_HOME':str(r/'home/.config'),'TMPDIR':str(r/'tmp')}
exe=out/'opt/nvidia/nsight-compute/2026.2.1/ncu';assert exe.is_file()
result={}
for flag,name in [('--version','version'),('--help','help')]:
 q=subprocess.run([str(exe),flag],env=env,cwd=r,capture_output=True,text=True,timeout=20);(r/(name+'.txt')).write_text(q.stdout+q.stderr);q.check_returncode();result[name]=q.stdout if name=='version' else len(q.stdout)
assert hashlib.sha256(Path('/var/lib/dpkg/status').read_bytes()).hexdigest()==before
result.update(state='EXTRACTED',files=count,unpacked_bytes=size,dpkg_unchanged=True,executable=str(exe));(r/'tool-status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
