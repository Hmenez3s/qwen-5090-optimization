from pathlib import Path
r=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-build-014')
p=r/'build-status.json';print(p.read_text() if p.exists() else 'NO_BUILD_STATUS')
base=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/build-ninfer-014')
for name in ['stdout.log','stderr.log']:
 p=base/name
 if p.exists():print(name,'\n'.join(p.read_text().splitlines()[-14:]))
