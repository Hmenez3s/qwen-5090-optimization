from pathlib import Path
import importlib.metadata as md
base='/home/operator/.venvs/qwen38/lib/python3.12/site-packages'
for d in md.distributions(path=[base]):
 if d.metadata['Name']=='nvidia-cutlass-dsl':
  for p in d.files or []:
   if 'env' in str(p).lower() and str(p).endswith('.py'):
    f=d.locate_file(p);print('FILE',f)
    for i,l in enumerate(f.read_text().splitlines(),1):
     if any(x in l for x in ['CACHE','cache','TMP','tmp','HOME']):print(i,l[:220])
