from pathlib import Path
import ast
p=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/managers/schedule_batch.py');s=p.read_text();t=ast.parse(s)
for n in ast.walk(t):
 if isinstance(n,ast.FunctionDef) and n.name in ['_new_tokens_required_next_decode_spec_v2']:
  print('FUNCTION',n.name,n.lineno,'\n'+'\n'.join(s.splitlines()[n.lineno-1:n.end_lineno]))
