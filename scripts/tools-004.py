from pathlib import Path
import subprocess,json
cuda=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/nvidia/cu13')
print('CUDA BIN',[(p.name,p.is_symlink()) for p in (cuda/'bin').iterdir()] if (cuda/'bin').exists() else 'missing')
for exe,args in [(str(cuda/'bin/nvcc'),['--version']),('/usr/bin/pkg-config',['--modversion','libavformat','libavcodec','libavutil','libswscale','libcurl']),('/usr/bin/g++',['--version'])]:
 try:
  r=subprocess.run([exe]+args,capture_output=True,text=True,timeout=10);print(exe,r.returncode,(r.stdout+r.stderr)[:2000])
 except Exception as e:print(exe,str(e))
for root in [Path('/home/operator/Qwen3.8-27B-NVFP4-RTX-5090'),Path('/home/operator/.venvs/qwen38/bin')]:
 if root.exists():print('ROOT',str(root),'FILES',[p.name for p in root.iterdir() if root.name!='bin' or p.name in ['cmake','ninja','nvcc','python','uv']])
print('ISOLATION',Path('/proc/sys/kernel/unprivileged_userns_clone').read_text() if Path('/proc/sys/kernel/unprivileged_userns_clone').exists() else 'no sysctl')
