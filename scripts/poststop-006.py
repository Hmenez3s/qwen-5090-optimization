from pathlib import Path
import json,subprocess
r=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908/sglang-baseline')
m=json.loads((r/'recovery-manifest.json').read_text())
print('original_processes_remaining',[(x['pid'],(Path('/proc')/str(x['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]) for x in m['children'] if (Path('/proc')/str(x['pid'])).exists()])
print(subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv'],capture_output=True,text=True).stdout)
print(subprocess.run(['ss','-ltnp'],capture_output=True,text=True).stdout)
