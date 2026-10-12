from pathlib import Path
p=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py')
print('\n'.join(p.read_text().splitlines()[2370:2535]))
