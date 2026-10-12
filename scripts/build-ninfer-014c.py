from pathlib import Path
import json,os,subprocess,time,hashlib
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-build-014')
CUDA=ROOT/'venv/lib/python3.12/site-packages/nvidia/cu13'
SRC=ROOT/'source/ninfer-b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868'
BUILD=ROOT/'build-c'
assert json.loads((ROOT/'status.json').read_text())['stage']=='SETUP_COMPLETE'
assert not BUILD.exists() and json.loads((ROOT/'build-status.json').read_text())['stage']=='FAILED'
assert hashlib.sha256((ROOT/'source.tar.gz').read_bytes()).hexdigest()=='866f3acbc21f2a9c2cc90fbed49ec4e7cd14b5890126ac0ac3bcca31c2af11f1'
link=CUDA/'lib/libcudart.so'
if not link.exists():link.symlink_to('libcudart.so.13')
env={'PATH':str(ROOT/'venv/bin')+':'+str(CUDA/'bin')+':/usr/bin:/bin','HOME':str(ROOT/'home'),'TMPDIR':'/tmp/qwen-build014','CUDA_HOME':str(CUDA),'CUDACXX':str(CUDA/'bin/nvcc'),'LD_LIBRARY_PATH':str(ROOT/'sysroot/usr/lib/x86_64-linux-gnu')+':'+str(CUDA/'lib')+':/usr/lib/wsl/lib','PKG_CONFIG_SYSROOT_DIR':str(ROOT/'sysroot'),'PKG_CONFIG_LIBDIR':str(ROOT/'sysroot/usr/lib/x86_64-linux-gnu/pkgconfig')+':'+str(ROOT/'sysroot/usr/lib/pkgconfig')+':'+str(ROOT/'sysroot/usr/share/pkgconfig'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','GIT_CONFIG_NOSYSTEM':'1','GIT_TERMINAL_PROMPT':'0'}
env['LIBRARY_PATH']=str(CUDA/'lib')+':/usr/lib/wsl/lib'
configure=[str(ROOT/'venv/bin/cmake'),'-S',str(SRC),'-B',str(BUILD),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=120a','-DCMAKE_CUDA_COMPILER='+str(CUDA/'bin/nvcc'),'-DCUDAToolkit_ROOT='+str(CUDA),'-DPKG_CONFIG_EXECUTABLE='+str(ROOT/'sysroot/usr/bin/pkgconf'),'-DCMAKE_LIBRARY_PATH=/usr/lib/wsl/lib;'+str(CUDA/'lib')+';'+str(ROOT/'sysroot/usr/lib/x86_64-linux-gnu'),'-DNINFER_BUILD_APPS=ON','-DBUILD_TESTING=OFF','-DNINFER_BUILD_BENCHMARKS=OFF']
commands=[configure,[str(ROOT/'venv/bin/cmake'),'--build',str(BUILD),'--parallel','2','--target','ninfer','ninfer-serve']]
(ROOT/'build-manifest.json').write_text(json.dumps({'commands':commands,'env':env},indent=2))
(ROOT/'build-pid.json').write_text(json.dumps({'pid':os.getpid(),'stat':Path('/proc/self/stat').read_text(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
start=time.monotonic()
try:
 for stage,cmd in zip(['CONFIGURING','COMPILING'],commands):
  (ROOT/'build-status.json').write_text(json.dumps({'stage':stage}));print('COMMAND',json.dumps(cmd),flush=True)
  subprocess.run(cmd,cwd=ROOT,env=env,check=True,timeout=max(1,3600-(time.monotonic()-start)))
 binaries={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BUILD/'apps/ninfer',BUILD/'apps/ninfer-serve']}
 (ROOT/'build-status.json').write_text(json.dumps({'stage':'COMPLETE','seconds':time.monotonic()-start,'binaries':binaries}))
except Exception as e:
 (ROOT/'build-status.json').write_text(json.dumps({'stage':'FAILED','error':repr(e),'seconds':time.monotonic()-start}));raise
