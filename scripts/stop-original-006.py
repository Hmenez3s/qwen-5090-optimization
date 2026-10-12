from pathlib import Path
import json,os,signal
r=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908/sglang-baseline')
m=json.loads((r/'recovery-manifest.json').read_text());p=Path('/proc')/str(m['original_pid'])
assert (p/'stat').read_text().split()[21]==m['original_stat'].split()[21], 'identity changed'
assert [x for x in (p/'cmdline').read_bytes().decode().split(chr(0)) if x]==m['original_argv'], 'argv changed'
os.kill(m['original_pid'],signal.SIGTERM)
print('SIGTERM sent to verified original SGLang PID',m['original_pid'])
