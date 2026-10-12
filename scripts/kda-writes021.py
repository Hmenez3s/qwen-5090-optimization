from pathlib import Path
import ast
p=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py');s=p.read_text();print('BYTES',len(s));t=ast.parse(s)
for n in t.body:
 if isinstance(n,(ast.Import,ast.ImportFrom)):print(ast.get_source_segment(s,n))
for i,l in enumerate(s.splitlines(),1):
 if any(x in l.lower() for x in ['cache','open(','path','mkdir','subprocess','requests','download','compile(']):print(i,l[:220])
