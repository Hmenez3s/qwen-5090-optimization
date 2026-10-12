from pathlib import Path
r=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/managers')
p=r/'tokenizer_control_mixin.py';print('CONTROL','\n'.join(p.read_text().splitlines()[379:412]))
p=r/'scheduler.py'
for i,l in enumerate(p.read_text().splitlines(),1):
 if 'profil' in l.lower():print('SCHEDULER',i,l)
