from pathlib import Path
p=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/build-ninfer-014/stderr.log')
print('\n'.join(p.read_text().splitlines()[105:205]))
