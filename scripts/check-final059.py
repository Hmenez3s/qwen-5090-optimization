#!/usr/bin/env python3
"""Read-only health check for the validated pi-serving-059 service."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from remote import run  # noqa: E402

EXPECTED = {
    "served_model_name": "qwen3.8-27b-normal",
    "context_length": 131200,
    "max_total_num_tokens": 131328,
    "attention_backend": "flashinfer",
    "speculative_algorithm": "DFLASH",
    "speculative_dflash_block_size": 8,
    "speculative_accept_threshold_single": 1.0,
    "speculative_accept_threshold_acc": 1.0,
    "fp4_gemm_runner_backend": "flashinfer_cutlass",
}


def get_json(url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return json.loads(response.read())


def main() -> int:
    try:
        local = get_json("http://127.0.0.1:8002/get_server_info")
        for key, value in EXPECTED.items():
            if local.get(key) != value:
                raise RuntimeError(
                    f"tunnel response mismatch: {key}={local.get(key)!r}, expected {value!r}"
                )

        remote_source = r'''from pathlib import Path
import hashlib,json,subprocess,urllib.request
root=Path('/home/operator/qwen38-5090/optimization-runs/pi-serving-059')
receipt=json.loads((root/'server-pid.json').read_text())
proc=Path('/proc')/str(receipt['pid'])
assert proc.is_dir(), 'recorded Linux PID is not alive'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==receipt['boot_id'], 'boot identity changed'
assert (proc/'stat').read_text().split()[21]==receipt['stat'].split()[21], 'PID was reused'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
server=json.loads(opener.open('http://127.0.0.1:8081/get_server_info',timeout=10).read())
expected={'served_model_name':'qwen3.8-27b-normal','context_length':131200,'max_total_num_tokens':131328,'attention_backend':'flashinfer','speculative_algorithm':'DFLASH','speculative_dflash_block_size':8,'speculative_accept_threshold_single':1.0,'speculative_accept_threshold_acc':1.0,'fp4_gemm_runner_backend':'flashinfer_cutlass'}
for key,value in expected.items():assert server.get(key)==value,(key,server.get(key),value)
manifest=json.loads((root/'prebuilt-aot.json').read_text());catalog=Path(manifest['directory'])
assert len(manifest['sha256'])==9
for rel,digest in manifest['sha256'].items():assert hashlib.sha256((catalog/rel).read_bytes()).hexdigest()==digest,rel
assert (root/'sitecustomize.py').read_bytes()==(root/'sitecustomize-kda059.py').read_bytes()
assert "added={(8,17408,5120),(8,5120,248320)}" in (root/'sitecustomize.py').read_text()
power=float(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).strip())
assert power==460.0,power
print(json.dumps({'status':'PASS','root':str(root),'pid':receipt['pid'],'model':server['served_model_name'],'context':server['context_length'],'pool':server['max_total_num_tokens'],'dflash_block':server['speculative_dflash_block_size'],'aot_hashes':9,'power_limit_W':power}))
'''
        result = run(remote_source, 60)
        if result.returncode:
            raise RuntimeError(
                "remote check failed:\n"
                + result.stdout.decode(errors="replace")
                + result.stderr.decode(errors="replace")
            )
        remote_status = json.loads(result.stdout)
        print(json.dumps({"status": "PASS", "tunnel": "8002 -> 8081", **remote_status}, indent=2))
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
