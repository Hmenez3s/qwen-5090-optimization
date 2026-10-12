from pathlib import Path
base=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages')
for p in base.glob('*cutlass*'):
 print(p)
 if p.is_dir() and not p.name.endswith('dist-info'):
  for q in p.glob('**/*env*.py'):print('ENV_SOURCE',q)
