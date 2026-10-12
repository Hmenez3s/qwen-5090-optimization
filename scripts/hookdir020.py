from pathlib import Path
r=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/ops/gemm')
print('DIR',r.is_dir())
if r.is_dir():
 for p in sorted(r.iterdir()):print(p.name,p.stat().st_size)
 p=r/'__init__.py'
 if p.exists():print(p.read_text())
