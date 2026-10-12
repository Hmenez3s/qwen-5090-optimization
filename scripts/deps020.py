from pathlib import Path
import importlib.metadata as md,subprocess,json
base=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages')
for d in md.distributions(path=[str(base)]):
 n=d.metadata['Name']
 if n and any(k in n.lower() for k in ['flashinfer','cublas','cudnn','cutlass','torch']):print(n,d.version)
print(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.limit','--format=csv,noheader'],text=True))
