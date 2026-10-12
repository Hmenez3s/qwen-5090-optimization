from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import micro_gemm022 as base
import micro_gemm027 as cache_audit


RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/micro029")
TMP_ROOT = Path("/tmp/qwen-micro029")
SNAPSHOT_SHA256 = "323a571e80018dc50f0c45593caff5240fb6dcd908f94af633b92f8896894586"
SHAPES = ((8, 17408, 5120), (8, 5120, 34816))
SEEDS = (17, 29, 43)
FLUSH_BYTES = 256 * 1024 * 1024
GRAPH_PREHEAT_SECONDS = 1.0
GRAPH_PREHEAT_BATCH = 8
TIMING_SAMPLES = 50
EXPECTED_CUTLASS = {(17408, 5120): 4, (5120, 34816): 2}
STOCK_TACTICS = tuple(
    (
        tile,
        (1, 1),
        False,
        use_prefetch,
        "sm120",
        None,
    )
    for tile in ((64, 64), (64, 128), (128, 64), (128, 128))
    for use_prefetch in (False, True)
)
REGIMES = (("l2_cold", FLUSH_BYTES, False), ("reuse", 0, True))


CacheMatchFailure = cache_audit.CacheMatchFailure
_tensor_receipt = cache_audit._tensor_receipt
_audit_search_cache = cache_audit._audit_search_cache
_require_cutlass_match = cache_audit._require_cutlass_match


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock-tactic and cache-regime b12x FP4 GEMM microbenchmark."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-cache", required=True)
    return parser.parse_args(argv)


def _validate_server_cache(args: argparse.Namespace) -> Path:
    path = Path(args.server_cache).expanduser()
    if not path.is_absolute():
        raise RuntimeError("--server-cache must be an absolute path")
    path = path.resolve()
    if not base._under(path, RUN_ROOT) or not path.is_file():
        raise RuntimeError(f"--server-cache must be an existing file under {RUN_ROOT}")
    if path.stat().st_mode & 0o222:
        raise RuntimeError(f"--server-cache must be read-only: {path}")
    return path


def _guard_environment() -> None:
    base._guard_environment(RUN_ROOT, TMP_ROOT)
    owned_cache_vars = (
        "CUDA_CACHE_PATH",
        "CUTE_DSL_CACHE_DIR",
        "CUTE_DSL_DUMP_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "SGLANG_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    )
    for name in owned_cache_vars:
        raw = os.environ.get(name)
        if not raw:
            raise RuntimeError(f"{name} must be set; no fallback is permitted")
        value = Path(raw).expanduser().resolve()
        if not base._under(value, RUN_ROOT):
            raise RuntimeError(f"{name} must be owned under {RUN_ROOT}, got {value}")
        if not value.is_dir():
            raise RuntimeError(f"{name} must name an existing directory: {value}")


def _json_tactic(tactic: tuple[Any, ...]) -> list[Any]:
    return [list(tactic[0]), list(tactic[1]), tactic[2], tactic[3], tactic[4], tactic[5]]


def _make_implementations(
    torch: Any,
    flashinfer: Any,
    device: Any,
    m: int,
    n: int,
    k: int,
    aq: Any,
    bq: Any,
    asf: Any,
    bsf: Any,
    alpha: Any,
    b12x_runner: Any,
    kernel_cache_module: Any,
    source_hashes: dict[str, Any],
    run_kda: Callable[..., Any] | None,
) -> tuple[dict[str, Callable[[], Any]], dict[str, dict[str, Any]]]:
    public_alpha = alpha.reshape(())
    receipts: dict[str, dict[str, Any]] = {}

    def reference() -> Any:
        return flashinfer.mm_fp4(
            aq, bq.T, asf, bsf.T, public_alpha, torch.bfloat16, backend="cutlass"
        )

    implementations: dict[str, Callable[[], Any]] = {
        "reference_cutlass_cache_matched": reference
    }
    valid_inputs: list[Any] | None = None
    for index, tactic in enumerate(STOCK_TACTICS):
        name = (
            f"b12x_stock_{tactic[0][0]}x{tactic[0][1]}_"
            f"{'prefetch' if tactic[3] else 'no_prefetch'}"
        )
        output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        if (
            tuple(output.shape) != (m, n)
            or output.dtype is not torch.bfloat16
            or not output.is_contiguous()
        ):
            raise RuntimeError(f"invalid persistent b12x output for {name}")
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        inputs = [
            aq,
            bq.T,
            asf,
            bsf.T,
            public_alpha,
            torch.bfloat16,
            output,
            16,
            True,
            workspace,
        ]
        if valid_inputs is None:
            valid_inputs = inputs
        # _compile_block_scaled_gemm namespaces the tactic by CUDA device.
        key = (torch.cuda.current_device(), (16, *tactic, True, torch.bfloat16))

        def make_direct(
            inputs: list[Any],
            tactic: tuple[Any, ...],
            name: str,
            key: tuple[Any, ...],
            output: Any,
            index: int,
        ) -> Callable[[], Any]:
            def run_direct() -> Any:
                b12x_runner(inputs=inputs, tactic=tactic)
                if key not in kernel_cache_module._B12X_MM_FP4_KERNEL_CACHE:
                    raise CacheMatchFailure(
                        f"b12x tactic {name} did not populate in-memory kernel cache key {key!r}"
                    )
                receipts[name] = {
                    "runner": b12x_runner.__class__.__name__,
                    "tactic": _json_tactic(tactic),
                    "key": repr(key),
                    "key_exists": True,
                    "source_hashes": source_hashes,
                    "ordinal": index,
                }
                return output

            return run_direct

        implementations[name] = make_direct(inputs, tactic, name, key, output, index)

    if valid_inputs is None:
        raise RuntimeError("stock tactic set is empty")
    valid_tactics = b12x_runner.get_valid_tactics(valid_inputs, None)
    missing = [tactic for tactic in STOCK_TACTICS if tactic not in valid_tactics]
    if missing:
        raise CacheMatchFailure(f"stock b12x tactics rejected by get_valid_tactics: {missing!r}")

    if run_kda is not None:
        implementations["kda_down"] = lambda: run_kda(aq, bq.T, asf, bsf.T, alpha)
    return implementations, receipts


def _timing_case(
    torch: Any,
    device: Any,
    m: int,
    n: int,
    k: int,
    seed: int,
    implementations: dict[str, Callable[[], Any]],
    tactic_receipts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for regime, flush_bytes, prime_reuse in REGIMES:
        bundles = base._capture_graphs(
            torch,
            device,
            implementations,
            flush_bytes=flush_bytes,
            prime_reuse=prime_reuse,
        )
        names = list(bundles)
        preheat_started = time.perf_counter()
        preheat_batches = 0
        preheat_replays = 0
        while True:
            for batch_index in range(GRAPH_PREHEAT_BATCH):
                order = names[(batch_index + preheat_batches) % len(names) :] + names[: (batch_index + preheat_batches) % len(names)]
                for name in order:
                    bundles[name]["graph"].replay()
                    preheat_replays += 1
            torch.cuda.synchronize(device)
            preheat_batches += 1
            if time.perf_counter() - preheat_started >= GRAPH_PREHEAT_SECONDS:
                break
        preheat_elapsed = time.perf_counter() - preheat_started

        samples: dict[str, list[float]] = {name: [] for name in names}
        for sample_index in range(TIMING_SAMPLES):
            offset = sample_index % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                bundle = bundles[name]
                bundle["graph"].replay()
                torch.cuda.synchronize(device)
                elapsed = float(bundle["start"].elapsed_time(bundle["end"]))
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError(f"invalid CUDA event time for {regime}/{name}: {elapsed}")
                samples[name].append(elapsed)

        for name, values in samples.items():
            tactic = tactic_receipts.get(name, {}).get("tactic")
            if tactic is None:
                tactic = EXPECTED_CUTLASS[(k, n)] if name.startswith("reference_") else "kda-down"
            records.append(
                {
                    "record_type": "timing",
                    "m": m,
                    "n": n,
                    "k": k,
                    "seed": seed,
                    "implementation": name,
                    "tactic": tactic,
                    "regime": regime,
                    "median_ms": float(statistics.median(values)),
                    "raw_samples_ms": values,
                    "samples_ms": values,
                    "sample_count": len(values),
                    "preheat_wall_seconds": preheat_elapsed,
                    "preheat_batches": preheat_batches,
                    "preheat_replays": preheat_replays,
                    "graph_protocol": {
                        "external_events": True,
                        "flush_bytes": flush_bytes,
                        "prime_reuse": prime_reuse,
                        "timed_call_between_events": True,
                        "order_rotation": "sample_index modulo implementation count",
                        "reuse_caveat": "not guaranteed fully resident in L2; up operand footprint is about 100MB versus 96MiB cache",
                    },
                    "claim_scope": "synthetic CUDA-graph timing only; not E2E or occupancy proof",
                }
            )
        del bundles
    return records


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    output = base._validate_output(args, RUN_ROOT)
    server_cache = _validate_server_cache(args)
    handle = output.open("x", encoding="utf-8")
    context: dict[str, Any] = {"stage": "guards"}
    timing_started = False
    timing_records = 0
    try:
        _guard_environment()
        context = {"stage": "imports"}
        import torch
        import flashinfer
        import flashinfer.autotuner.autotuner as autotuner_module
        import flashinfer.fp4_quantization as flashinfer_compat_quantization
        import flashinfer.gemm.gemm_base as flashinfer_gemm_base
        import flashinfer.gemm.gemm_mm_fp4_cute_dsl as cute_dsl_module
        import flashinfer.quantization.fp4_quantization as flashinfer_quantization
        import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm as kda_gate
        import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120 as kda_sm120
        import sglang.srt.layers.quantization.fp4_utils as sglang_fp4_utils
        import torch.cuda.graphs as torch_graphs
        import torch.cuda.streams as torch_streams
        from sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120 import (
            _run_qwen3x_nvfp4_gemm,
        )

        device, properties, free_bytes = base._device_gate(torch)
        cache_data = json.loads(server_cache.read_text(encoding="utf-8"))
        clone_receipt = base._hash_file(server_cache)
        if clone_receipt["sha256"] != SNAPSHOT_SHA256:
            raise CacheMatchFailure("server cache differs from pinned snapshot")
        snapshot_data = cache_data
        cache_hashes = {"server_clone": clone_receipt, "snapshot_sha256": SNAPSHOT_SHA256}
        source_modules = {
            "flashinfer.autotuner": autotuner_module,
            "flashinfer.gemm.gemm_base": flashinfer_gemm_base,
            "flashinfer.gemm.gemm_mm_fp4_cute_dsl": cute_dsl_module,
            "flashinfer.fp4_quantization_compat": flashinfer_compat_quantization,
            "flashinfer.quantization.fp4_quantization": flashinfer_quantization,
            "sglang.kda_gate": kda_gate,
            "sglang.kda_sm120": kda_sm120,
            "sglang.fp4_utils": sglang_fp4_utils,
            "torch.cuda.graphs": torch_graphs,
            "torch.cuda.streams": torch_streams,
        }
        source_files = {
            name: base._hash_file(base._module_file(module))
            for name, module in source_modules.items()
        }
        source_files["micro_gemm022"] = base._hash_file(Path(base.__file__).resolve())
        source_files["micro_gemm029"] = base._hash_file(Path(__file__).resolve())
        b12x_source_hashes = dict(source_files)
        base._emit(
            handle,
            {
                "record_type": "metadata",
                "experiment": "micro_gemm029",
                "status": "STARTED",
                "shapes": [{"m": m, "k": k, "n": n} for m, k, n in SHAPES],
                "seeds": list(SEEDS),
                "candidates": [
                    "reference_cutlass_cache_matched",
                    *[
                        f"b12x_stock_{t[0][0]}x{t[0][1]}_{'prefetch' if t[3] else 'no_prefetch'}"
                        for t in STOCK_TACTICS
                    ],
                    "kda_down",
                ],
                "cache": {"hashes": cache_hashes},
                "device": "cuda:0",
                "gpu": {
                    "name": str(properties.name),
                    "major": int(properties.major),
                    "minor": int(properties.minor),
                    "total_memory": int(properties.total_memory),
                    "free_memory_before_allocations": free_bytes,
                },
                "versions": {
                    "python": sys.version,
                    "torch": str(torch.__version__),
                    "torch_cuda": str(torch.version.cuda),
                    "flashinfer": base._version("flashinfer", flashinfer),
                    "sglang": base._version("sglang", kda_gate),
                },
                "source_files": source_files,
                "numerical_gate": {
                    "normalized_l2_max": 1e-3,
                    "elementwise": "absdiff <= 0.015625 * max(abs(reference), 1)",
                    "finite_required": True,
                    "bitwise_fraction_reported": True,
                },
                "timing_protocol": {
                    "graphs_per_implementation": True,
                    "regimes": [
                        {"name": name, "flush_bytes": flush, "prime_reuse": prime}
                        for name, flush, prime in REGIMES
                    ],
                    "external_events": True,
                    "preheat_seconds": GRAPH_PREHEAT_SECONDS,
                    "samples_per_implementation": TIMING_SAMPLES,
                    "interleaved_with_rotating_order": True,
                    "reuse_caveat": "not guaranteed fully resident in L2; up operand footprint is about 100MB versus 96MiB cache",
                },
                "claim_scope": "synthetic benchmark only; no hardware occupancy, model quality, or E2E claim",
            },
        )

        tuner = autotuner_module.AutoTuner.get()
        validated: list[tuple[int, int, int, int, dict[str, Callable[[], Any]], dict[str, dict[str, Any]]]] = []
        if not tuner.load_configs(str(server_cache)):
            raise CacheMatchFailure("server cache metadata rejected")
        numerical_failures: list[dict[str, Any]] = []
        with autotuner_module.autotune(False):
            for m, k, n in SHAPES:
                for seed in SEEDS:
                    context = {"stage": "numerical", "m": m, "k": k, "n": n, "seed": seed}
                    _, _, aq, bq, asf, bsf, alpha, a_global, b_global = base._quantize_case(
                        torch, flashinfer, device, m, seed, k=k, n=n
                    )
                    b12x_runner = flashinfer_gemm_base._b12x_gemm_fp4_runner(
                        12, 0, True, torch.bfloat16, True
                    )
                    implementations, tactic_receipts = _make_implementations(
                        torch,
                        flashinfer,
                        device,
                        m,
                        n,
                        k,
                        aq,
                        bq,
                        asf,
                        bsf,
                        alpha,
                        b12x_runner,
                        flashinfer_gemm_base,
                        b12x_source_hashes,
                        _run_qwen3x_nvfp4_gemm if (m, k, n) == (8, 17408, 5120) else None,
                    )
                    reference_output, initial_observations = _audit_search_cache(
                        tuner, implementations["reference_cutlass_cache_matched"], autotuner_module
                    )
                    initial_match = _require_cutlass_match(
                        initial_observations,
                        cache_data,
                        snapshot_data,
                        k,
                        n,
                        "before_stock_tactics",
                    )
                    base._emit(handle, {"record_type": "cache_match", **initial_match, "m": m, "k": k, "n": n, "seed": seed, "source_hashes": source_files})

                    candidates: dict[str, Any] = {}
                    for name in implementations:
                        if name.startswith("b12x_stock_"):
                            output_value = implementations[name]()
                            candidates[name] = output_value
                            base._emit(
                                handle,
                                {
                                    "record_type": "b12x_tactic",
                                    "m": m,
                                    "k": k,
                                    "n": n,
                                    "seed": seed,
                                    "implementation": name,
                                    **tactic_receipts[name],
                                    "output": _tensor_receipt(output_value),
                                },
                            )
                    if "kda_down" in implementations:
                        candidates["kda_down"] = implementations["kda_down"]()
                    _, final_observations = _audit_search_cache(
                        tuner, implementations["reference_cutlass_cache_matched"], autotuner_module
                    )
                    final_match = _require_cutlass_match(
                        final_observations,
                        cache_data,
                        snapshot_data,
                        k,
                        n,
                        "after_stock_tactics",
                    )
                    base._emit(handle, {"record_type": "cache_match", **final_match, "m": m, "k": k, "n": n, "seed": seed, "source_hashes": source_files})

                    base._emit(
                        handle,
                        {
                            "record_type": "operands",
                            "m": m,
                            "k": k,
                            "n": n,
                            "seed": seed,
                            "a": _tensor_receipt(aq),
                            "b_transposed_input": _tensor_receipt(bq.T),
                            "a_scale": _tensor_receipt(asf),
                            "b_scale_transposed_input": _tensor_receipt(bsf.T),
                            "alpha_public_scalar": _tensor_receipt(alpha.reshape(())),
                            "alpha_kda": _tensor_receipt(alpha),
                            "a_global_scale": a_global,
                            "b_global_scale": b_global,
                        },
                    )
                    reference_for_gate = reference_output
                    case_failed = False
                    for name, candidate in candidates.items():
                        passed, details = base._failure_details(torch, reference_for_gate, candidate)
                        record = {
                            "record_type": "numerical",
                            "m": m,
                            "k": k,
                            "n": n,
                            "seed": seed,
                            "implementation": name,
                            "reference": "reference_cutlass_cache_matched",
                            "passed": passed,
                            **details,
                        }
                        base._emit(handle, record)
                        if not passed:
                            case_failed = True
                            numerical_failures.append(record)
                    if not case_failed:
                        validated.append((m, k, n, seed, implementations, tactic_receipts))

        if numerical_failures:
            raise base.NumericalGateFailure(
                f"{len(numerical_failures)} numerical candidate failures: "
                + json.dumps(
                    [
                        {
                            "m": item["m"],
                            "k": item["k"],
                            "n": item["n"],
                            "seed": item["seed"],
                            "implementation": item["implementation"],
                        }
                        for item in numerical_failures
                    ],
                    sort_keys=True,
                )
            )

        for m, k, n, seed, implementations, tactic_receipts in validated:
            context = {"stage": "timing", "m": m, "k": k, "n": n, "seed": seed}
            timing_started = True
            for record in _timing_case(
                torch, device, m, n, k, seed, implementations, tactic_receipts
            ):
                base._emit(handle, record)
                timing_records += 1

        base._emit(
            handle,
            {
                "record_type": "numerical_gate",
                "status": "PASS",
                "cases": len(SHAPES) * len(SEEDS),
                "timings_reported_only_after_gate": True,
            },
        )
        base._emit(
            handle,
            {
                "record_type": "summary",
                "experiment": "micro_gemm029",
                "status": "PASS",
                "final_status": "PASS",
                "timing_records": timing_records,
                "claim_scope": "synthetic tolerance and CUDA-graph timing only; not model quality, E2E speed, or occupancy proof",
            },
        )
        return 0
    except base.NumericalGateFailure as exc:
        base._emit(
            handle,
            {
                "record_type": "summary",
                "experiment": "micro_gemm029",
                "status": "FAIL",
                "final_status": "FAIL",
                "reason": str(exc),
                "timings_reported": False,
                "claim_scope": "no timing or promotion claim",
            },
        )
        return 2
    except Exception as exc:
        base._emit(
            handle,
            {
                "record_type": "error",
                "experiment": "micro_gemm029",
                "status": "ERROR",
                "final_status": "ERROR",
                "stage": context,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timings_reported": timing_started and timing_records > 0,
            },
        )
        return 1
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
