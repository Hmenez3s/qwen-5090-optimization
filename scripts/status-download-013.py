from pathlib import Path
r=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-resume-013')
print((r/'status.json').read_text() if (r/'status.json').exists() else 'NO_STATUS')
