from pathlib import Path
for name,r in [('download',Path('/home/operator/qwen38-5090/optimization-runs/ninfer-resume-013')),('setup',Path('/home/operator/qwen38-5090/optimization-runs/ninfer-build-014'))]:
 p=r/'status.json';print(name,p.read_text() if p.exists() else 'NO_STATUS')
log=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/install-ninfer-014/stderr.log')
if log.exists():print('SETUP_STDERR','\n'.join(log.read_text().splitlines()[-10:]))
