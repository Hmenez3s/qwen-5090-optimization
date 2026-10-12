from pathlib import Path
import subprocess
print(subprocess.run(['ps','-eo','pid,ppid,etimes,pcpu,rss,comm','--sort=-pcpu'],capture_output=True,text=True).stdout[:4500])
print(subprocess.run(['free','-m'],capture_output=True,text=True).stdout)
root=Path('/tmp/qwen-r007-20260908T211616')
for p in sorted(root.glob('**/build.ninja')): print(p)
