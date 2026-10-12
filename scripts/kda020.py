from pathlib import Path
import ast
p=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py');s=p.read_text();t=ast.parse(s);print('SIZE',len(s))
for n in t.body:
 if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign)) or isinstance(n,ast.FunctionDef) and n.name=='try_qwen3x_nvfp4_gemm':
  text=ast.get_source_segment(s,n)
  if len(text)<18000:print(text)
