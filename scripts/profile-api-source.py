from pathlib import Path
import ast
base=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/managers')
for filename,names in [('io_struct.py',['ProfileReq']),('tokenizer_manager.py',['start_profile','stop_profile'])]:
 p=base/filename;s=p.read_text();t=ast.parse(s)
 for n in ast.walk(t):
  if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names:
   print(filename,n.name,n.lineno,'\n'+'\n'.join(s.splitlines()[n.lineno-1:n.end_lineno]))
