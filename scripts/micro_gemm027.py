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


RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/micro027")
TMP_ROOT = Path("/tmp/qwen-micro027")
SNAPSHOT_SHA256 = "323a571e80018dc50f0c45593caff5240fb6dcd908f94af633b92f8896894586"
SHAPES = ((8, 17408, 5120), (8, 5120, 34816))
SEEDS = (17, 29, 43)
FLUSH_BYTES = 256 * 1024 * 1024
GRAPH_PREHEAT_SECONDS = 1.0
GRAPH_PREHEAT_BATCH = 8
TIMING_SAMPLES = 50
EXPECTED_CUTLASS = {(17408, 5120): 4, (5120, 34816): 2}


class CacheMatchFailure(RuntimeError):
    pass


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache-matched CUTLASS versus tuned b12x/KDA FP4 GEMM microbenchmark."
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
    for name in ("CUDA_CACHE_PATH", "CUTE_DSL_CACHE_DIR", "CUTE_DSL_DUMP_DIR"):
        value = Path(os.environ[name]).expanduser().resolve()
        if not base._under(value, RUN_ROOT):
            raise RuntimeError(f"{name} must be owned under {RUN_ROOT}, got {value}")


def _tensor_receipt(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "strides": list(tensor.stride()),
        "device": str(tensor.device),
    }


def _audit_search_cache(
    tuner: Any, call: Callable[[], Any], autotuner_module: Any
) -> tuple[Any, list[dict[str, Any]]]:
    original = tuner.search_cache
    observations: list[dict[str, Any]] = []

    def audited(
        custom_op: str,
        runners: list[Any],
        input_shapes: tuple[tuple[int, ...], ...],
        tuning_config: Any,
        inputs: list[Any] | None = None,
    ) -> tuple[bool, int, Any, Any]:
        result = original(custom_op, runners, input_shapes, tuning_config, inputs=inputs)
        if custom_op == "fp4_gemm":
            hit, runner_id, tactic, _ = result
            runner = runners[runner_id]
            extras = runner.get_cache_key_extras(inputs) if inputs is not None else ()
            key = autotuner_module.AutoTuner._get_cache_key(
                custom_op, runner, input_shapes, tuning_config, extras
            )
            observations.append(
                {
                    "hit": bool(hit),
                    "runner": runner.__class__.__name__,
                    "tactic": tactic if isinstance(tactic, int) else repr(tactic),
                    "file_key": key.file_key,
                }
            )
        return result

    tuner.search_cache = audited
    try:
        return call(), observations
    finally:
        tuner.search_cache = original


def _require_cutlass_match(
    observations: list[dict[str, Any]],
    cache_data: dict[str, Any],
    snapshot_data: dict[str, Any],
    k: int,
    n: int,
    phase: str,
) -> dict[str, Any]:
    expected_tactic = EXPECTED_CUTLASS[(k, n)]
    matches = [
        item
        for item in observations
        if item["runner"] == "CutlassFp4GemmRunner"
    ]
    if not matches:
        raise CacheMatchFailure(f"{phase}: no CutlassFp4GemmRunner search_cache observation")
    receipt = matches[-1]
    if not receipt["hit"] or receipt["tactic"] != expected_tactic:
        raise CacheMatchFailure(f"{phase}: expected cache hit Cutlass tactic {expected_tactic}, got {receipt}")
    key = receipt["file_key"]
    for label, data in (("server clone", cache_data), ("supplied snapshot", snapshot_data)):
        record = data.get(key)
        if record != ["CutlassFp4GemmRunner", expected_tactic]:
            raise CacheMatchFailure(f"{phase}: file_key missing or mismatched in {label}: {key}")
    return {"phase": phase, **receipt, "snapshot_record": snapshot_data[key]}


def _make_implementations(
    torch: Any,
    flashinfer: Any,
    aq: Any,
    bq: Any,
    asf: Any,
    bsf: Any,
    alpha: Any,
    run_kda: Callable[..., Any] | None,
) -> dict[str, Callable[[], Any]]:
    public_alpha = alpha.reshape(())

    def reference() -> Any:
        return flashinfer.mm_fp4(
            aq, bq.T, asf, bsf.T, public_alpha, torch.bfloat16, backend="cutlass"
        )

    def b12x() -> Any:
        return flashinfer.mm_fp4(
            aq, bq.T, asf, bsf.T, public_alpha, torch.bfloat16, backend="b12x"
        )

    implementations: dict[str, Callable[[], Any]] = {
        "reference_cutlass_cache_matched": reference,
        "b12x_tuned": b12x,
    }
    if run_kda is not None:
        implementations["kda_down"] = lambda: run_kda(aq, bq.T, asf, bsf.T, alpha)
    return implementations


def _timing_case(
    torch: Any,
    device: Any,
    m: int,
    n: int,
    k: int,
    seed: int,
    implementations: dict[str, Callable[[], Any]],
) -> list[dict[str, Any]]:
    bundles = base._capture_graphs(torch, device, implementations)
    started = time.perf_counter()
    batches = 0
    replays = 0
    while True:
        for _ in range(GRAPH_PREHEAT_BATCH):
            for bundle in bundles.values():
                bundle["graph"].replay()
                replays += 1
        torch.cuda.synchronize(device)
        batches += 1
        if time.perf_counter() - started >= GRAPH_PREHEAT_SECONDS:
            break
    preheat_elapsed = time.perf_counter() - started

    samples: dict[str, list[float]] = {name: [] for name in bundles}
    for _ in range(TIMING_SAMPLES):
        for name, bundle in bundles.items():
            bundle["graph"].replay()
            torch.cuda.synchronize(device)
            elapsed = float(bundle["start"].elapsed_time(bundle["end"]))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"invalid CUDA event time for {name}: {elapsed}")
            samples[name].append(elapsed)

    records = []
    for name, values in samples.items():
        records.append(
            {
                "record_type": "timing",
                "m": m,
                "n": n,
                "k": k,
                "seed": seed,
                "implementation": name,
                "samples_ms": values,
                "median_ms": float(statistics.median(values)),
                "sample_count": len(values),
                "l2_cold_flush_bytes": FLUSH_BYTES,
                "preheat_wall_seconds": preheat_elapsed,
                "preheat_batches": batches,
                "preheat_replays": replays,
                "timing": "CUDA external events recorded inside graph after flush",
            }
        )
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
        source_files["micro_gemm027"] = base._hash_file(Path(__file__).resolve())
        base._emit(
            handle,
            {
                "record_type": "metadata",
                "experiment": "micro_gemm027",
                "status": "STARTED",
                "shapes": [{"m": m, "k": k, "n": n} for m, k, n in SHAPES],
                "seeds": list(SEEDS),
                "candidates": ["reference_cutlass_cache_matched", "b12x_tuned", "kda_down"],
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
                    "flush_bytes": FLUSH_BYTES,
                    "external_events": True,
                    "preheat_seconds": GRAPH_PREHEAT_SECONDS,
                    "samples_per_implementation": TIMING_SAMPLES,
                    "interleaved": True,
                },
            },
        )

        tuner = autotuner_module.AutoTuner.get()
        validated: list[tuple[int, int, int, int, dict[str, Callable[[], Any]]]] = []
        if not tuner.load_configs(str(server_cache)):
            raise CacheMatchFailure("server cache metadata rejected")
        with autotuner_module.autotune(False):
            for m, k, n in SHAPES:
                for seed in SEEDS:
                    context = {"stage": "numerical", "m": m, "k": k, "n": n, "seed": seed}
                    _, _, aq, bq, asf, bsf, alpha, a_global, b_global = base._quantize_case(
                        torch, flashinfer, device, m, seed, k=k, n=n
                    )
                    implementations = _make_implementations(
                        torch, flashinfer, aq, bq, asf, bsf, alpha,
                        _run_qwen3x_nvfp4_gemm
                        if (m, k, n) == (8, 17408, 5120)
                        else None,
                    )
                    reference_output, initial_observations = _audit_search_cache(
                        tuner, implementations["reference_cutlass_cache_matched"], autotuner_module
                    )
                    initial_match = _require_cutlass_match(
                        initial_observations, cache_data, snapshot_data, k, n, "before_b12x_tuning"
                    )
                    base._emit(handle, {"record_type": "cache_match", **initial_match, "m": m, "k": k, "n": n, "seed": seed})

                    with autotuner_module.autotune(True):
                        implementations["b12x_tuned"]()
                    b12x_output, b12x_observations = _audit_search_cache(
                        tuner, implementations["b12x_tuned"], autotuner_module
                    )
                    b12x_matches = [
                        item for item in b12x_observations
                        if item["runner"] == "B12xFp4GemmRunner" and item["hit"]
                    ]
                    if not b12x_matches:
                        raise CacheMatchFailure(f"after_b12x_tuning: no in-memory B12x cache hit for {(m, k, n)}")
                    base._emit(handle, {"record_type": "tuned_selection", **b12x_matches[-1], "m": m, "k": k, "n": n, "seed": seed})

                    reference_output, final_observations = _audit_search_cache(
                        tuner, implementations["reference_cutlass_cache_matched"], autotuner_module
                    )
                    final_match = _require_cutlass_match(
                        final_observations, cache_data, snapshot_data, k, n, "after_b12x_tuning"
                    )
                    base._emit(handle, {"record_type": "cache_match", **final_match, "m": m, "k": k, "n": n, "seed": seed})

                    operand_record = {
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
                    }
                    base._emit(handle, operand_record)
                    candidates = {"b12x_tuned": b12x_output}
                    if "kda_down" in implementations:
                        candidates["kda_down"] = implementations["kda_down"]()
                    failure = None
                    for name, candidate in candidates.items():
                        passed, details = base._failure_details(torch, reference_output, candidate)
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
                        if not passed and failure is None:
                            failure = record
                    if failure is not None:
                        raise base.NumericalGateFailure(
                            f"numerical gate failed for M={m}, K={k}, N={n}, seed={seed}, "
                            f"implementation={failure['implementation']}"
                        )
                    validated.append((m, k, n, seed, implementations))

            for m, k, n, seed, implementations in validated:
                context = {"stage": "timing", "m": m, "k": k, "n": n, "seed": seed}
                timing_started = True
                for record in _timing_case(torch, device, m, n, k, seed, implementations):
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
                "experiment": "micro_gemm027",
                "status": "PASS",
                "final_status": "PASS",
                "timing_records": timing_records,
                "claim_scope": "synthetic tolerance and CUDA-graph timing only; not model quality or E2E speed proof",
            },
        )
        return 0
    except base.NumericalGateFailure as exc:
        base._emit(handle, {"record_type": "summary", "experiment": "micro_gemm027", "status": "FAIL", "final_status": "FAIL", "reason": str(exc), "timings_reported": False, "claim_scope": "no timing or promotion claim"})
        return 2
    except Exception as exc:
        base._emit(handle, {"record_type": "error", "experiment": "micro_gemm027", "status": "ERROR", "final_status": "ERROR", "stage": context, "error_type": type(exc).__name__, "error": str(exc), "timings_reported": timing_started and timing_records > 0})
        return 1
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
