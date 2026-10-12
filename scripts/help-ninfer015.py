from pathlib import Path
import json,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-build-014')
m=json.loads((r/'build-manifest.json').read_text())
p=subprocess.run([str(r/'build-c/apps/ninfer-serve'),'--help'],cwd=r,env=m['env'],capture_output=True,text=True,timeout=20)
print(p.stdout);print(p.stderr);p.check_returncode()
