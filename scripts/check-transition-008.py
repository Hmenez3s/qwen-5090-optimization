from pathlib import Path
import json
r=Path('/home/operator/qwen38-5090/optimization-runs/exp008-dflash8');m=json.loads((r/'launch.json').read_text());p=Path('/proc')/str(m['wait_for']['pid'])
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==m['wait_for']['boot_id']
assert (p/'stat').read_text().split()[21]==m['wait_for']['start_ticks']
assert [x for x in (p/'cmdline').read_bytes().decode().split(chr(0)) if x]==json.loads((r/'original-argv.json').read_text())
log=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/experiment-008-dflash8/stdout.log').read_text()
assert '"stage": "waiting"' in log
print('IDENTITY_AND_NEW_CONTROLLER_WAIT_VERIFIED')
