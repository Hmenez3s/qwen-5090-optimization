from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/micro022")
TMP_ROOT = Path("/tmp/qwen-micro022")
MIN_FREE_BYTES = 27000 * 1024 * 1024
FLUSH_BYTES = 256 * 1024 * 1024
M = (8, 9)
SEEDS = (17, 29, 43)
K = 17408
N = 5120
WARMUP_CALLS = 3
GRAPH_PREHEAT_SECONDS = 1.0
GRAPH_PREHEAT_BATCH = 8
TIMING_SAMPLES = 50


class NumericalGateFailure(RuntimeError):
    pass


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic SM120 NVFP4 KDA microbenchmark for K17408/N5120."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new JSONL output under /home/operator/qwen38-5090/optimization-runs/micro022/",
    )
    return parser.parse_args(argv)


def _validate_output(
    args: argparse.Namespace, run_root: Path = RUN_ROOT
) -> Path:
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        raise RuntimeError("--output must be an absolute path")
    output = output.resolve()
    if output.suffix != ".jsonl" or not _under(output, run_root):
        raise RuntimeError(f"--output must be a new .jsonl under {run_root}")
    if output == run_root or not output.parent.is_dir():
        raise RuntimeError("--output parent must already exist inside the run root")
    if output.exists():
        raise RuntimeError(f"refusing existing exclusive output: {output}")
    return output


def _guard_environment(
    run_root: Path = RUN_ROOT, tmp_root: Path = TMP_ROOT
) -> None:
    cwd = Path.cwd().resolve()
    if not _under(cwd, run_root):
        raise RuntimeError(f"cwd must be inside {run_root}, got {cwd}")

    env_roots = {
        "HOME": (run_root,),
        "TMPDIR": (tmp_root,),
        "CUDA_CACHE_PATH": (run_root, tmp_root),
        "CUTE_DSL_CACHE_DIR": (run_root, tmp_root),
        "CUTE_DSL_DUMP_DIR": (run_root, tmp_root),
    }
    for name, roots in env_roots.items():
        raw = os.environ.get(name)
        if not raw:
            raise RuntimeError(f"{name} must be set; no fallback is permitted")
        value = Path(raw).expanduser()
        if not value.is_absolute():
            raise RuntimeError(f"{name} must be an absolute path")
        value = value.resolve()
        if not any(_under(value, root) for root in roots):
            allowed = ", ".join(str(root) for root in roots)
            raise RuntimeError(f"{name}={value} is outside allowed roots: {allowed}")
        if not value.is_dir():
            raise RuntimeError(f"{name} must name an existing directory: {value}")


def _emit(handle: Any, record: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value
    handle.write(json.dumps(safe(record), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
    handle.flush()


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _module_file(module: Any) -> Path:
    filename = getattr(module, "__file__", None)
    if not filename:
        raise RuntimeError(f"module has no source file: {module!r}")
    path = Path(filename).resolve()
    if not path.is_file():
        raise RuntimeError(f"module source file is not a regular file: {path}")
    return path


def _version(package: str, module: Any) -> str:
    candidates = (package, "flashinfer-python") if package == "flashinfer" else (package,)
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            pass
    return str(getattr(module, "__version__", "unknown"))


def _device_gate(torch: Any) -> tuple[Any, Any, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device0 is unavailable")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    capability = (int(properties.major), int(properties.minor))
    if capability != (12, 0):
        raise RuntimeError(f"device0 must be SM120, got SM{capability[0]}{capability[1]}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(
            f"device0 needs at least {MIN_FREE_BYTES} free bytes, got {free_bytes}"
        )
    return device, properties, int(free_bytes)


def _global_scale(torch: Any, tensor: Any) -> float:
    maximum = float(tensor.abs().max().item())
    if not maximum or not maximum == maximum:
        raise RuntimeError("synthetic operand has invalid or zero abs.max()")
    return 2688.0 / maximum


def _quantize_case(
    torch: Any,
    flashinfer: Any,
    device: Any,
    m: int,
    seed: int,
    k: int = K,
    n: int = N,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=generator)
    b = torch.randn((n, k), device=device, dtype=torch.bfloat16, generator=generator)
    a_global = _global_scale(torch, a)
    b_global = _global_scale(torch, b)
    a_global_tensor = torch.tensor(
        [a_global], device=device, dtype=torch.float32
    )
    b_global_tensor = torch.tensor(
        [b_global], device=device, dtype=torch.float32
    )
    layout = flashinfer.SfLayout.layout_128x4
    aq, asf = flashinfer.nvfp4_quantize(
        a,
        a_global_tensor,
        sfLayout=layout,
        do_shuffle=False,
        backend="cuda",
    )
    bq, bsf = flashinfer.nvfp4_quantize(
        b,
        b_global_tensor,
        sfLayout=layout,
        do_shuffle=False,
        backend="cuda",
    )
    alpha = torch.tensor(
        [1.0 / (a_global * b_global)], device=device, dtype=torch.float32
    )
    return a, b, aq, bq, asf, bsf, alpha, a_global, b_global


def _make_implementations(
    torch: Any,
    flashinfer: Any,
    device: Any,
    m: int,
    aq: Any,
    bq: Any,
    asf: Any,
    bsf: Any,
    alpha: Any,
    compile_decode_kernel: Callable[..., Any],
    run_kda: Callable[..., Any],
) -> dict[str, Callable[[], Any]]:
    def reference() -> Any:
        return flashinfer.mm_fp4(
            aq,
            bq.T,
            asf,
            bsf.T,
            alpha,
            torch.bfloat16,
            backend="cutlass",
        )

    def kda_existing_run() -> Any:
        return run_kda(aq, bq.T, asf, bsf.T, alpha)

    implementations: dict[str, Callable[[], Any]] = {
        "reference_cutlass": reference,
        "kda_existing_run": kda_existing_run,
    }

    if m == 8:
        sf_m = (m + 127) // 128
        sf_n = (N + 127) // 128
        sf_k = ((K // 16) + 3) // 4
        kernel = compile_decode_kernel(
            0,
            sf_m,
            sf_n,
            sf_k,
            direct_scheduler=N < K,
            m1_epilogue=False,
            cache_policy=True,
        )

        def kda_m8_cache_true() -> Any:
            output = torch.empty(
                m, N, device=device, dtype=torch.bfloat16
            )
            kernel(
                aq,
                bq,
                output,
                sf_m,
                sf_n,
                sf_k,
                asf.data_ptr(),
                bsf.T.data_ptr(),
                alpha.reshape(1),
            )
            return output

        implementations["kda_m8_cache_true"] = kda_m8_cache_true
    return implementations


def _failure_details(torch: Any, reference: Any, candidate: Any) -> tuple[bool, dict[str, Any]]:
    finite = bool(torch.isfinite(reference).all().item()) and bool(
        torch.isfinite(candidate).all().item()
    )
    diff = (candidate.float() - reference.float()).abs()
    ref_float = reference.float()
    ref_norm = torch.linalg.vector_norm(ref_float)
    l2 = float((torch.linalg.vector_norm(diff) / ref_norm.clamp_min(1e-12)).item())
    max_abs = float(diff.max().item())
    bitwise_equal_fraction = float(
        (candidate.view(torch.int16) == reference.view(torch.int16)).float().mean().item()
    )
    allowed = 0.015625 * torch.maximum(
        ref_float.abs(), torch.ones_like(ref_float)
    )
    tolerance_mask = diff <= allowed
    tolerance_pass = bool(tolerance_mask.all().item())
    l2_pass = l2 <= 1e-3
    passed = finite and tolerance_pass and l2_pass
    details: dict[str, Any] = {
        "finite": finite,
        "normalized_l2_error": l2,
        "normalized_l2_limit": 1e-3,
        "max_abs_error": max_abs,
        "elementwise_tolerance": "0.015625 * max(abs(reference), 1)",
        "elementwise_tolerance_pass": tolerance_pass,
        "normalized_l2_pass": l2_pass,
        "bitwise_equal_fraction": bitwise_equal_fraction,
    }
    if not finite or not tolerance_pass or not l2_pass:
        bad = (~tolerance_mask).nonzero(as_tuple=False)
        if bad.numel():
            index = tuple(int(item) for item in bad[0].tolist())
            details["first_elementwise_failure"] = {
                "index": list(index),
                "reference": float(reference[index].item()),
                "candidate": float(candidate[index].item()),
                "abs_error": float(diff[index].item()),
                "allowed": float(allowed[index].item()),
            }
        if not finite:
            details["nonfinite_reference"] = int(
                (~torch.isfinite(reference)).sum().item()
            )
            details["nonfinite_candidate"] = int(
                (~torch.isfinite(candidate)).sum().item()
            )
    return passed, details


def _numerical_case(
    torch: Any,
    flashinfer: Any,
    device: Any,
    m: int,
    seed: int,
    compile_decode_kernel: Callable[..., Any],
    run_kda: Callable[..., Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, tuple[Any, ...] | None]:
    (
        a,
        b,
        aq,
        bq,
        asf,
        bsf,
        alpha,
        a_global,
        b_global,
    ) = _quantize_case(torch, flashinfer, device, m, seed)
    implementations = _make_implementations(
        torch,
        flashinfer,
        device,
        m,
        aq,
        bq,
        asf,
        bsf,
        alpha,
        compile_decode_kernel,
        run_kda,
    )
    with torch.no_grad():
        reference = implementations["reference_cutlass"]()
        candidates = {
            name: implementation()
            for name, implementation in implementations.items()
            if name != "reference_cutlass"
        }
    torch.cuda.synchronize(device)

    records: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    for name, candidate in candidates.items():
        passed, details = _failure_details(torch, reference, candidate)
        record = {
            "record_type": "numerical",
            "m": m,
            "n": N,
            "k": K,
            "seed": seed,
            "implementation": name,
            "reference": "reference_cutlass",
            "a_global_scale": a_global,
            "b_global_scale": b_global,
            "passed": passed,
            **details,
        }
        records.append(record)
        if not passed and failure is None:
            failure = record

    del candidates, reference, implementations, a, b
    if failure is not None:
        del aq, bq, asf, bsf, alpha
        return records, failure, None
    return records, None, (aq, bq, asf, bsf, alpha)


def _capture_graphs(
    torch: Any,
    device: Any,
    implementations: dict[str, Callable[[], Any]],
) -> dict[str, dict[str, Any]]:
    with torch.no_grad():
        for implementation in implementations.values():
            for _ in range(WARMUP_CALLS):
                output = implementation()
                del output
        torch.cuda.synchronize(device)

        bundles: dict[str, dict[str, Any]] = {}
        for name, implementation in implementations.items():
            flush = torch.empty(
                FLUSH_BYTES // 4, device=device, dtype=torch.float32
            )
            graph = torch.cuda.CUDAGraph()
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            try:
                # The documented context manager supplies a non-default capture
                # stream and capture_end instantiates keep_graph=False graphs.
                with torch.cuda.graph(graph):
                    flush.zero_()
                    start.record()
                    output = implementation()
                    end.record()
            except Exception as exc:
                raise RuntimeError(
                    "CUDA graph capture with external timing events is unsupported "
                    f"or failed for {name}: {exc}"
                ) from exc
            bundles[name] = {
                "graph": graph,
                "flush": flush,
                "start": start,
                "end": end,
                "output": output,
            }
        torch.cuda.synchronize(device)
    return bundles


def _timing_case(
    torch: Any,
    flashinfer: Any,
    device: Any,
    m: int,
    seed: int,
    case_tensors: tuple[Any, ...],
    compile_decode_kernel: Callable[..., Any],
    run_kda: Callable[..., Any],
) -> list[dict[str, Any]]:
    aq, bq, asf, bsf, alpha = case_tensors
    implementations = _make_implementations(
        torch,
        flashinfer,
        device,
        m,
        aq,
        bq,
        asf,
        bsf,
        alpha,
        compile_decode_kernel,
        run_kda,
    )
    bundles = _capture_graphs(torch, device, implementations)

    preheat_started = time.perf_counter()
    preheat_batches = 0
    preheat_replays = 0
    while True:
        for _ in range(GRAPH_PREHEAT_BATCH):
            for bundle in bundles.values():
                bundle["graph"].replay()
                preheat_replays += 1
        torch.cuda.synchronize(device)
        preheat_batches += 1
        if time.perf_counter() - preheat_started >= GRAPH_PREHEAT_SECONDS:
            break

    preheat_elapsed = time.perf_counter() - preheat_started
    samples: dict[str, list[float]] = {name: [] for name in bundles}
    for _ in range(TIMING_SAMPLES):
        for name, bundle in bundles.items():
            bundle["graph"].replay()
            torch.cuda.synchronize(device)
            elapsed_ms = float(bundle["start"].elapsed_time(bundle["end"]))
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise RuntimeError(f"invalid CUDA event time for {name}: {elapsed_ms}")
            samples[name].append(elapsed_ms)

    records = []
    for name, values in samples.items():
        records.append(
            {
                "record_type": "timing",
                "m": m,
                "n": N,
                "k": K,
                "seed": seed,
                "implementation": name,
                "samples_ms": values,
                "median_ms": float(statistics.median(values)),
                "sample_count": len(values),
                "l2_cold_flush_bytes": FLUSH_BYTES,
                "preheat_wall_seconds": preheat_elapsed,
                "preheat_batches": preheat_batches,
                "preheat_replays": preheat_replays,
                "timing": "CUDA external events recorded inside graph after flush",
            }
        )

    del bundles, implementations, aq, bq, asf, bsf, alpha
    return records


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    output = _validate_output(args)
    handle = output.open("x", encoding="utf-8")
    context: dict[str, Any] = {"stage": "guards"}
    try:
        _guard_environment()
        context = {"stage": "imports"}
        import torch
        import flashinfer
        import flashinfer.fp4_quantization as flashinfer_compat_quantization
        import flashinfer.gemm.gemm_base as flashinfer_gemm_base
        import flashinfer.quantization.fp4_quantization as flashinfer_quantization
        import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm as kda_gate
        import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120 as kda_sm120
        import sglang.srt.layers.quantization.fp4_utils as sglang_fp4_utils
        import torch.cuda.graphs as torch_graphs
        import torch.cuda.streams as torch_streams
        from sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120 import (
            _compile_decode_kernel,
            _run_qwen3x_nvfp4_gemm,
        )

        device, properties, free_bytes = _device_gate(torch)
        source_modules = {
            "sglang.kda_gate": kda_gate,
            "sglang.kda_sm120": kda_sm120,
            "flashinfer.gemm.gemm_base": flashinfer_gemm_base,
            "flashinfer.fp4_quantization_compat": flashinfer_compat_quantization,
            "flashinfer.quantization.fp4_quantization": flashinfer_quantization,
            "sglang.fp4_utils": sglang_fp4_utils,
            "torch.cuda.graphs": torch_graphs,
            "torch.cuda.streams": torch_streams,
        }
        _emit(
            handle,
            {
                "record_type": "metadata",
                "experiment": "micro_gemm022",
                "status": "STARTED",
                "shapes": {"m": list(M), "n": N, "k": K},
                "seeds": list(SEEDS),
                "device": "cuda:0",
                "gpu": {
                    "name": str(properties.name),
                    "major": int(properties.major),
                    "minor": int(properties.minor),
                    "multi_processor_count": int(properties.multi_processor_count),
                    "total_memory": int(properties.total_memory),
                    "free_memory_before_allocations": free_bytes,
                },
                "versions": {
                    "python": sys.version,
                    "torch": str(torch.__version__),
                    "torch_cuda": str(torch.version.cuda),
                    "flashinfer": _version("flashinfer", flashinfer),
                    "sglang": _version("sglang", kda_gate),
                },
                "source_files": {
                    name: _hash_file(_module_file(module))
                    for name, module in source_modules.items()
                },
                "numerical_gate": {
                    "normalized_l2_max": 1e-3,
                    "elementwise": "absdiff <= 0.015625 * max(abs(reference), 1)",
                    "finite_required": True,
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

        timing_records: list[dict[str, Any]] = []
        validated_cases: list[tuple[int, int, tuple[Any, ...]]] = []
        for m in M:
            for seed in SEEDS:
                context = {"stage": "numerical", "m": m, "seed": seed}
                numerical_records, failure, case_tensors = _numerical_case(
                    torch,
                    flashinfer,
                    device,
                    m,
                    seed,
                    _compile_decode_kernel,
                    _run_qwen3x_nvfp4_gemm,
                )
                for record in numerical_records:
                    _emit(handle, record)
                if failure is not None:
                    raise NumericalGateFailure(
                        f"numerical gate failed for M={m}, seed={seed}, "
                        f"implementation={failure['implementation']}"
                    )

                assert case_tensors is not None
                validated_cases.append((m, seed, case_tensors))

        for m, seed, case_tensors in validated_cases:
            context = {"stage": "timing", "m": m, "seed": seed}
            timing_records.extend(
                _timing_case(
                    torch, flashinfer, device, m, seed, case_tensors,
                    _compile_decode_kernel, _run_qwen3x_nvfp4_gemm,
                )
            )

        _emit(
            handle,
            {
                "record_type": "numerical_gate",
                "status": "PASS",
                "cases": len(M) * len(SEEDS),
                "timings_reported_only_after_gate": True,
            },
        )
        for record in timing_records:
            _emit(handle, record)
        _emit(
            handle,
            {
                "record_type": "summary",
                "experiment": "micro_gemm022",
                "status": "PASS",
                "claim_scope": "synthetic tolerance and CUDA-graph timing only; not model quality proof",
            },
        )
        return 0
    except NumericalGateFailure as exc:
        _emit(
            handle,
            {
                "record_type": "summary",
                "experiment": "micro_gemm022",
                "status": "FAIL",
                "reason": str(exc),
                "timings_reported": False,
                "claim_scope": "no timing or promotion claim",
            },
        )
        return 2
    except Exception as exc:
        _emit(
            handle,
            {
                "record_type": "error",
                "experiment": "micro_gemm022",
                "status": "ERROR",
                "stage": context,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timings_reported": False,
            },
        )
        return 1
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
