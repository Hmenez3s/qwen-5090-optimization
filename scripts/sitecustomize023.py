"""EXP023 process-local eligibility extension; installed source remains unchanged."""
import hashlib
import os
import sys
from pathlib import Path

try:
    import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm as gate
    expected = 'b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5'
    if hashlib.sha256(Path(gate.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('EXP023 gate source hash mismatch')
    if (8, 17408, 5120) in gate._SUPPORTED_SHAPES:
        raise RuntimeError('EXP023 baseline eligibility changed')
    gate._SUPPORTED_SHAPES = gate._SUPPORTED_SHAPES.union({(8, 17408, 5120)})
    print('EXP023_KDA_M8_ENABLED pid=' + str(os.getpid()), file=sys.stderr, flush=True)
except Exception as exc:
    print('EXP023_PATCH_FAILED: ' + repr(exc), file=sys.stderr, flush=True)
    # Python normally ignores sitecustomize errors; never run an unpatched trial.
    os._exit(78)
