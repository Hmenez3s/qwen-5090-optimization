from __future__ import annotations

from pathlib import Path

import micro_gemm029 as bench029


EXPERIMENT = "micro_gemm037"
PROFILE_SOURCE = Path(__file__).resolve()
RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/micro037")
TMP_ROOT = Path("/tmp/qwen-micro037")
SHAPES = ((8, 5120, 248320),)
SNAPSHOT_SHA256 = "678beee4fe0b33f95a2613bf1b7ff3c3705e68e8faad114836377c09fdd6ec68"
KDA_SHAPES = {(8, 5120, 248320): "kda_lm_head_private"}
REUSE_CAVEAT = (
    "Head weights plus scales ~682.03 MiB exceed 96 MiB L2; immediate reuse is not "
    "full residency or an E2E claim."
)
EXPECTED_CUTLASS = {(5120, 248320): 6}


bench029.EXPERIMENT = EXPERIMENT
bench029.PROFILE_SOURCE = PROFILE_SOURCE
bench029.RUN_ROOT = RUN_ROOT
bench029.TMP_ROOT = TMP_ROOT
bench029.SHAPES = SHAPES
bench029.SNAPSHOT_SHA256 = SNAPSHOT_SHA256
bench029.KDA_SHAPES = KDA_SHAPES
bench029.REUSE_CAVEAT = REUSE_CAVEAT
bench029.EXPECTED_CUTLASS = EXPECTED_CUTLASS
bench029.cache_audit.EXPECTED_CUTLASS = EXPECTED_CUTLASS


if __name__ == "__main__":
    raise SystemExit(bench029.main())
