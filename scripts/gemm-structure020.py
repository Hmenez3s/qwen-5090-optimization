from pathlib import Path
base=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages')
for rel in ['sglang/srt/layers/quantization','flashinfer/gemm']:
 r=base/rel;print('ROOT',rel)
 if r.exists():
  for p in sorted(r.iterdir()):print(p.name,'dir' if p.is_dir() else p.stat().st_size)
print('FLASHINFER TOP GEMM FILES')
for p in (base/'flashinfer').glob('*gemm*'):print(p.name)
