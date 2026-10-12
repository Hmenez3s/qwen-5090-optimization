from __future__ import annotations

from pathlib import Path
from typing import Any

import micro_gemm029 as bench029


EXPERIMENT = "micro_gemm034"
PROFILE_SOURCE = Path(__file__).resolve()
RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/micro034")
TMP_ROOT = Path("/tmp/qwen-micro034")
STOCK_CONTROL_TACTIC = ((64, 128), (1, 1), False, False, "sm120", None)
NARROW_TACTICS = tuple(
    ((tile_m, tile_n), (1, 1), True, use_prefetch, "sm120", None)
    for tile_n in (16, 32)
    for use_prefetch in (False, True)
    for tile_m in (64,)
)
TACTICS = (STOCK_CONTROL_TACTIC, *NARROW_TACTICS)


def _validate_tactics(
    runner: Any, inputs: list[Any], tactics: tuple[tuple[Any, ...], ...]
) -> list[dict[str, Any]]:
    if tuple(tactics) != TACTICS:
        raise bench029.CacheMatchFailure(
            f"micro_gemm034 received undeclared tactics: {tactics!r}"
        )

    valid_tactics = runner.get_valid_tactics(inputs, None)
    admissions: list[dict[str, Any]] = []
    for ordinal, tactic in enumerate(tactics):
        if tactic == STOCK_CONTROL_TACTIC:
            success = tactic in valid_tactics
            receipt: dict[str, Any] = {
                "type": "runner.get_valid_tactics",
                "success": bool(success),
                "tactic": bench029._json_tactic(tactic),
                "ordinal": ordinal,
            }
        elif tactic in NARROW_TACTICS:
            import cutlass
            from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
                Sm120B12xBlockScaledDenseGemmKernel,
            )

            tile = tactic[0]
            public_n = int(inputs[1].shape[1])
            real_k = int(inputs[0].shape[1]) * 2
            success = bool(
                Sm120B12xBlockScaledDenseGemmKernel.can_implement(
                    cutlass.Float4E2M1FN,
                    cutlass.Float8E4M3FN,
                    16,
                    cutlass.BFloat16,
                    tile,
                    (1, 1),
                    public_n,
                    real_k,
                    1,
                    "k",
                    "k",
                    "n",
                    load_path="tma",
                    swap_ab=True,
                )
            )
            receipt = {
                "type": "Sm120B12xBlockScaledDenseGemmKernel.can_implement",
                "success": success,
                "tactic": bench029._json_tactic(tactic),
                "ordinal": ordinal,
                "arguments": {
                    "ab_dtype": "cutlass.Float4E2M1FN",
                    "sf_dtype": "cutlass.Float8E4M3FN",
                    "sf_vec_size": 16,
                    "c_dtype": "cutlass.BFloat16",
                    "mma_tiler_mn": list(tile),
                    "cluster_shape_mn": [1, 1],
                    "n": public_n,
                    "k": real_k,
                    "l": 1,
                    "a_major": "k",
                    "b_major": "k",
                    "c_major": "n",
                    "load_path": "tma",
                    "swap_ab": True,
                },
            }
        else:
            success = False
            receipt = {
                "type": "declared_tactic",
                "success": False,
                "tactic": bench029._json_tactic(tactic),
                "ordinal": ordinal,
            }

        if not success:
            raise bench029.CacheMatchFailure(
                f"micro_gemm034 tactic admission failed: {receipt!r}"
            )
        admissions.append(receipt)
    return admissions


bench029.EXPERIMENT = EXPERIMENT
bench029.PROFILE_SOURCE = PROFILE_SOURCE
bench029.RUN_ROOT = RUN_ROOT
bench029.TMP_ROOT = TMP_ROOT
bench029.TACTICS = TACTICS
bench029.TACTIC_VALIDATOR = _validate_tactics


if __name__ == "__main__":
    raise SystemExit(bench029.main())
