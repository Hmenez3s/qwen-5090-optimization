from pathlib import Path
p=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/env_manager.py')
s=p.read_text().splitlines()
for i,l in enumerate(s):
 if any(x in l.lower() for x in ['cache','dump_dir','tmp','home']):print(i+1,' | '.join(s[max(0,i-1):i+3]))
