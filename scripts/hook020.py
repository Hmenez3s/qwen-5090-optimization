from pathlib import Path
import ast
p=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/ops/gemm.py');s=p.read_text();t=ast.parse(s);print('SIZE',len(s))
for n in t.body:
 if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name=='try_qwen3x_nvfp4_gemm':print(ast.get_source_segment(s,n))
