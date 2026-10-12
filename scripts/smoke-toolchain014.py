from pathlib import Path
import os,subprocess,json
r=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-build-014');cuda=r/'venv/lib/python3.12/site-packages/nvidia/cu13'
env={'PATH':str(r/'venv/bin')+':'+str(cuda/'bin')+':/usr/bin:/bin','HOME':str(r/'home'),'TMPDIR':'/tmp/qwen-build014','LD_LIBRARY_PATH':str(r/'sysroot/usr/lib/x86_64-linux-gnu')+':'+str(cuda/'lib')+':/usr/lib/wsl/lib','PKG_CONFIG_SYSROOT_DIR':str(r/'sysroot'),'PKG_CONFIG_LIBDIR':str(r/'sysroot/usr/lib/x86_64-linux-gnu/pkgconfig')+':'+str(r/'sysroot/usr/lib/pkgconfig')+':'+str(r/'sysroot/usr/share/pkgconfig'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
for cmd in [[str(cuda/'bin/nvcc'),'--version'],[str(r/'venv/bin/cmake'),'--version'],[str(r/'venv/bin/ninja'),'--version'],[str(r/'sysroot/usr/bin/pkgconf'),'--modversion','libavformat','libavcodec','libavutil','libswscale','libcurl']]:
 p=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=r,timeout=15);print(json.dumps({'command':cmd,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}));p.check_returncode()
print('CUDA_LIBS',[p.name for p in (cuda/'lib').iterdir()][:20]);print('NVTX3',(cuda/'include/nvtx3').exists())
