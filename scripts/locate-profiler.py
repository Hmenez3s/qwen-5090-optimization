from pathlib import Path
r=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/managers')
for p in r.glob('*.py'):
 t=p.read_text()
 for i,l in enumerate(t.splitlines(),1):
  if any(k in l for k in ['def start_profile(', 'def stop_profile(', 'def _start_profile(', 'class SchedulerProfilerMixin']):print(p.name,i,l.strip())
