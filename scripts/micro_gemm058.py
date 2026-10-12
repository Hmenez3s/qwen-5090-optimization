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


EXPERIMENT = "micro_gemm058"
PROFILE_SOURCE = Path(__file__).resolve()
RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/micro058")
TMP_ROOT = Path("/tmp/qwen-micro058")
SNAPSHOT_SHA256 = "678beee4fe0b33f95a2613bf1b7ff3c3705e68e8faad114836377c09fdd6ec68"
M, K, N = 8, 5120, 34816
SHAPE = (M, K, N)
SHAPES = (SHAPE,)
SEEDS = (17, 29, 43)
MIN_FREE_BYTES = 27000 * 1024 * 1024
FLUSH_BYTES = 256 * 1024 * 1024
GRAPH_PREHEAT_SECONDS = 1.0
GRAPH_PREHEAT_BATCH = 8
TIMING_SAMPLES = 50
REFERENCE_NAME = "reference_cutlass_cache_tactic2"
EXPECTED_CUTLASS_TACTIC = 2
EXPECTED_CUTLASS_FILE_KEY = (
    "('fp4_gemm', 'CutlassFp4GemmRunner', ((8, 2560), (2560, 34816), "
    "(-1, 320), (320, 34816), (), (0,), (-1, 34816), (0,), (0,), (-1,)), ())"
)
DIRECT_SPECS = (
    ("direct_b12x_tma_prefetch_off", "tma", False),
    ("direct_b12x_tma_prefetch_on", "tma", True),
    ("direct_b12x_cpasync_prefetch_off", "cpasync", False),
    ("direct_b12x_cpasync_prefetch_on", "cpasync", True),
)
DIRECT_NAMES = tuple(item[0] for item in DIRECT_SPECS)
REGIMES = (("l2_cold", FLUSH_BYTES, False), ("reuse", 0, True))
REUSE_CAVEAT = (
    "The reuse regime is an immediate second graph call; it does not establish "
    "full L2 residency or an end-to-end claim."
)


class CacheMatchFailure(RuntimeError):
    pass


class NumericalGateFailure(RuntimeError):
    pass


class DirectAdmissionFailure(RuntimeError):
    pass


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct SM120 NVFP4 b12x tile-64x128 load-path microbenchmark "
            "against cache-matched CUTLASS."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new JSONL output under /home/operator/qwen38-5090/optimization-runs/micro058/",
    )
    parser.add_argument(
        "--server-cache",
        required=True,
        help="read-only cache clone under /home/operator/qwen38-5090/optimization-runs/micro058/",
    )
    return parser.parse_args(argv)


def _validate_output(args: argparse.Namespace) -> Path:
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        raise RuntimeError("--output must be an absolute path")
    output = output.resolve()
    if output.suffix != ".jsonl" or not _under(output, RUN_ROOT):
        raise RuntimeError(f"--output must be a new .jsonl under {RUN_ROOT}")
    if output == RUN_ROOT or not output.parent.is_dir():
        raise RuntimeError("--output parent must already exist inside the run root")
    if output.exists():
        raise RuntimeError(f"refusing existing exclusive output: {output}")
    return output


def _validate_server_cache(args: argparse.Namespace) -> Path:
    path = Path(args.server_cache).expanduser()
    if not path.is_absolute():
        raise RuntimeError("--server-cache must be an absolute path")
    path = path.resolve()
    if not _under(path, RUN_ROOT) or not path.is_file():
        raise RuntimeError(f"--server-cache must be an existing file under {RUN_ROOT}")
    if path.stat().st_mode & 0o222:
        raise RuntimeError(f"--server-cache must be read-only: {path}")
    return path


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

    owned_cache_vars = (
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
        if not _under(value, run_root):
            raise RuntimeError(f"{name} must be owned under {run_root}, got {value}")
        if not value.is_dir():
            raise RuntimeError(f"{name} must name an existing directory: {value}")


def _emit(handle: Any, record: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    handle.write(
        json.dumps(
            safe(record), sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    handle.flush()


def _hash_bytes(data: bytes, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


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


def _tensor_receipt(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "strides": list(tensor.stride()),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
    }


def _check_output(torch: Any, output: Any, m: int = M, n: int = N) -> None:
    if tuple(output.shape) != (m, n):
        raise RuntimeError(f"output shape must be {(m, n)}, got {tuple(output.shape)}")
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"output dtype must be torch.bfloat16, got {output.dtype}")
    if not output.is_contiguous():
        raise RuntimeError("output must be contiguous")


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
    free_bytes, _ = torch.cuda.mem_get_info(device)
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


def _quantize_case(torch: Any, flashinfer: Any, device: Any, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=generator)
    b = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=generator)
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
    bq_raw, bsf_raw = flashinfer.nvfp4_quantize(
        b,
        b_global_tensor,
        sfLayout=layout,
        do_shuffle=False,
        backend="cuda",
    )
    # Keep bq/bsf in the public mm_fp4 B orientation (K/2, N).  The direct
    # wrapper then receives the installed runner's kernel_b=bq.T and
    # kernel_b_sf=bsf.T orientations.
    bq = bq_raw.T
    bsf = bsf_raw.T
    alpha = torch.tensor(
        [1.0 / (a_global * b_global)], device=device, dtype=torch.float32
    )
    return a, b, aq, bq, asf, bsf, alpha, a_global, b_global


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
        result = original(
            custom_op, runners, input_shapes, tuning_config, inputs=inputs
        )
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


def _read_cache_pair(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snapshot_bytes = path.read_bytes()
    clone_bytes = path.read_bytes()
    if snapshot_bytes != clone_bytes:
        raise CacheMatchFailure("read-only server cache changed between snapshot and clone reads")
    snapshot_hash = _hash_bytes(snapshot_bytes, path)
    clone_hash = _hash_bytes(clone_bytes, path)
    if snapshot_hash["sha256"] != SNAPSHOT_SHA256:
        raise CacheMatchFailure(
            f"server cache sha256 differs from pinned snapshot: {snapshot_hash['sha256']}"
        )
    try:
        snapshot_data = json.loads(snapshot_bytes.decode("utf-8"))
        clone_data = json.loads(clone_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheMatchFailure(f"server cache is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(snapshot_data, dict) or not isinstance(clone_data, dict):
        raise CacheMatchFailure("server cache root must be a JSON object")
    return (
        clone_data,
        snapshot_data,
        {
            "server_clone": clone_hash,
            "supplied_snapshot": snapshot_hash,
            "pinned_sha256": SNAPSHOT_SHA256,
            "byte_equal": True,
            "read_only": True,
        },
    )


def _require_cutlass_match(
    observations: list[dict[str, Any]],
    clone_data: dict[str, Any],
    snapshot_data: dict[str, Any],
    cache_receipt: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in observations
        if item["runner"] == "CutlassFp4GemmRunner"
    ]
    if not matches:
        raise CacheMatchFailure(
            f"{phase}: no CutlassFp4GemmRunner search_cache observation"
        )
    receipt = matches[-1]
    if not receipt["hit"] or receipt["tactic"] != EXPECTED_CUTLASS_TACTIC:
        raise CacheMatchFailure(
            f"{phase}: expected cache hit Cutlass tactic {EXPECTED_CUTLASS_TACTIC}, "
            f"got {receipt}"
        )
    if receipt["file_key"] != EXPECTED_CUTLASS_FILE_KEY:
        raise CacheMatchFailure(
            f"{phase}: observed unexpected packed file_key {receipt['file_key']!r}"
        )
    for label, data in (
        ("server clone", clone_data),
        ("supplied snapshot", snapshot_data),
    ):
        record = data.get(EXPECTED_CUTLASS_FILE_KEY)
        if record != ["CutlassFp4GemmRunner", EXPECTED_CUTLASS_TACTIC]:
            raise CacheMatchFailure(
                f"{phase}: exact file_key missing or mismatched in {label}: "
                f"{EXPECTED_CUTLASS_FILE_KEY}"
            )
    return {
        "phase": phase,
        "runner": receipt["runner"],
        "hit": receipt["hit"],
        "tactic": receipt["tactic"],
        "file_key": receipt["file_key"],
        "clone_record": clone_data[EXPECTED_CUTLASS_FILE_KEY],
        "snapshot_record": snapshot_data[EXPECTED_CUTLASS_FILE_KEY],
        "cache": cache_receipt,
    }


def _direct_compile_key(
    torch: Any,
    cutlass: Any,
    load_path: str,
    use_prefetch: bool,
    sf_m: int,
    sf_n: int,
    sf_k: int,
) -> tuple[Any, ...]:
    # This is deliberately separate from FlashInfer's package cache.  Every
    # constructor-static field and every compile-static field is part of the
    # process-local key; _compile_block_scaled_gemm adds the device prefix.
    ctor_static = (
        ("sf_vec_size", 16),
        ("mma_tiler_mn", (64, 128)),
        ("cluster_shape_mn", (1, 1)),
        ("mma_k", 64),
        ("tile_k", None),
        ("single_work_tile_per_cta", False),
        ("use_prefetch", bool(use_prefetch)),
        ("enable_pdl", True),
        ("direct_one_m_tile_scheduler", False),
        ("split_k_slices", 1),
        ("split_k_atomic_bf16", False),
        ("use_m1_non_tma_a", False),
        ("use_m1_non_tma_c", False),
        ("use_m1_non_tma_sfa", False),
        ("load_path", load_path),
        ("swap_ab", False),
    )
    compile_static = (
        ("ab_logical_dtype", str(cutlass.Float4E2M1FN)),
        ("ab_cutlass_dtype", str(cutlass.Uint8)),
        ("sf_dtype", str(cutlass.Float8E4M3FN)),
        ("c_cutlass_dtype", str(cutlass.BFloat16)),
        ("out_dtype", str(torch.bfloat16)),
        ("acc_dtype", str(cutlass.Float32)),
        ("ab_assumed_align", 32),
        ("c_assumed_align", 16),
        ("sf_pointer_align", 16),
        ("alpha_dtype", str(cutlass.Float32)),
        ("alpha_assumed_align", 4),
        ("cluster_shape_mn", (1, 1)),
        ("cluster_shape_k", 1),
        ("swap_ab", False),
        ("kernel_type", "sm120"),
        ("use_tma_store", None),
        ("epilogue_op", "default_identity"),
        ("sf_m", sf_m),
        ("sf_n", sf_n),
        ("sf_k", sf_k),
        ("batch_size", 1),
        ("cache_module_name", None),
        ("compile_options", "--opt-level 2 --enable-tvm-ffi"),
    )
    return ("micro_gemm058_direct_b12x", ctor_static, compile_static)


def _make_implementations(
    torch: Any,
    flashinfer: Any,
    cutlass: Any,
    device: Any,
    m: int,
    n: int,
    k: int,
    aq: Any,
    bq: Any,
    asf: Any,
    bsf: Any,
    alpha: Any,
    direct_kernel_class: Any,
    compile_block_scaled_gemm: Callable[..., Any],
    prepare_alpha_for_launch: Callable[..., Any],
    direct_kernel_cache: dict[tuple[Any, ...], tuple[Any, Any]],
    device_index: int,
    source_hashes: dict[str, Any],
) -> tuple[dict[str, Callable[[], Any]], dict[str, dict[str, Any]]]:
    public_alpha = alpha.reshape(())
    kernel_a = aq
    kernel_b = bq.T
    kernel_a_sf = asf
    kernel_b_sf = bsf.T
    sf_m = (m + 127) // 128
    sf_n = (n + 127) // 128
    sf_k = (k // 16 + 3) // 4
    implementations: dict[str, Callable[[], Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}

    def reference() -> Any:
        output = flashinfer.mm_fp4(
            kernel_a,
            bq,
            kernel_a_sf,
            bsf,
            public_alpha,
            torch.bfloat16,
            backend="cutlass",
        )
        _check_output(torch, output, m, n)
        return output

    implementations[REFERENCE_NAME] = reference

    for name, load_path, use_prefetch in DIRECT_SPECS:
        # Keep this exact static admission call before construction or compile.
        can_implement = direct_kernel_class.can_implement(
            # The logical A/B type is FP4; compilation stores packed values as Uint8.
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            (64, 128),
            (1, 1),
            n,
            k,
            1,
            "k",
            "k",
            "n",
            load_path=load_path,
            swap_ab=False,
        )
        if not can_implement:
            raise DirectAdmissionFailure(
                f"{name}: static can_implement returned false for load_path={load_path}"
            )
        compile_key = _direct_compile_key(
            torch,
            cutlass,
            load_path,
            use_prefetch,
            sf_m,
            sf_n,
            sf_k,
        )
        stored_key = (device_index, compile_key)
        alpha_for_launch = prepare_alpha_for_launch(alpha, device)
        admission = {
            "call": "Sm120B12xBlockScaledDenseGemmKernel.can_implement",
            "ab_dtype": str(cutlass.Float4E2M1FN),
            "sf_dtype": str(cutlass.Float8E4M3FN),
            "sf_vec_size": 16,
            "c_dtype": str(cutlass.BFloat16),
            "mma_tiler_mn": (64, 128),
            "cluster_shape_mn": (1, 1),
            "n": n,
            "k": k,
            "l": 1,
            "majors": ("k", "k", "n"),
            "load_path": load_path,
            "swap_ab": False,
            "result": bool(can_implement),
        }
        compile_receipt: dict[str, Any] = {
            "process_local_cache": True,
            "cache_module_name": None,
            "device_index": device_index,
            "cache_key": compile_key,
            "stored_key": stored_key,
            "stored_key_has_device_prefix": stored_key[0] == device_index,
            "stored": stored_key in direct_kernel_cache,
        }
        state: dict[str, Any] = {"compiled": False, "last_output": None}
        receipt: dict[str, Any] = {
            "implementation": name,
            "load_path": load_path,
            "use_prefetch": bool(use_prefetch),
            "tile": (64, 128),
            "cluster": (1, 1),
            "enable_pdl": True,
            "swap_ab": False,
            "batch_size": 1,
            "can_implement": admission,
            "compile_key": compile_receipt,
            "alpha_for_launch": _tensor_receipt(alpha_for_launch),
            "source_hashes": dict(source_hashes),
        }
        receipts[name] = receipt

        def make_direct(
            name: str = name,
            load_path: str = load_path,
            use_prefetch: bool = use_prefetch,
            compile_key: tuple[Any, ...] = compile_key,
            stored_key: tuple[Any, ...] = stored_key,
            alpha_for_launch: Any = alpha_for_launch,
            receipt: dict[str, Any] = receipt,
            state: dict[str, Any] = state,
        ) -> Callable[[], Any]:
            def run_direct() -> Any:
                if not state["compiled"]:
                    cache_before = stored_key in direct_kernel_cache

                    def make_kernel() -> Any:
                        # Keep the direct constructor contract literal: all other
                        # constructor options remain their installed defaults.
                        return direct_kernel_class(
                            16,
                            (64, 128),
                            (1, 1),
                            use_prefetch=use_prefetch,
                            enable_pdl=True,
                            load_path=load_path,
                            swap_ab=False,
                        )

                    compiled_gemm, max_active_clusters = compile_block_scaled_gemm(
                        direct_kernel_cache,
                        compile_key,
                        make_kernel,
                        ab_cutlass_dtype=cutlass.Uint8,
                        sf_dtype=cutlass.Float8E4M3FN,
                        c_cutlass_dtype=cutlass.BFloat16,
                        ab_assumed_align=32,
                        cluster_shape_mn=(1, 1),
                        swap_ab=False,
                        sf_m=sf_m,
                        sf_n=sf_n,
                        sf_k=sf_k,
                        batch_size=1,
                        cluster_shape_k=1,
                        cache_module_name=None,
                        device_index=device_index,
                    )
                    if stored_key not in direct_kernel_cache:
                        raise DirectAdmissionFailure(
                            f"{name}: direct compile did not store device-prefixed key "
                            f"{stored_key!r}"
                        )
                    receipt["compile_key"].update(
                        {
                            "cache_before": cache_before,
                            "stored": True,
                            "max_active_clusters": int(max_active_clusters),
                        }
                    )
                    state["compiled"] = True
                    state["compiled_gemm"] = compiled_gemm

                output = torch.empty(
                    (m, n), device=device, dtype=torch.bfloat16
                )
                _check_output(torch, output, m, n)
                # These are the exact nine runtime arguments of the installed wrapper.
                state["compiled_gemm"](
                    kernel_a,
                    kernel_b,
                    output,
                    sf_m,
                    sf_n,
                    sf_k,
                    kernel_a_sf.data_ptr(),
                    kernel_b_sf.data_ptr(),
                    alpha_for_launch,
                )
                state["last_output"] = output
                receipt["last_output"] = _tensor_receipt(output)
                return output

            return run_direct

        implementations[name] = make_direct()

    return implementations, receipts


def _failure_details(
    torch: Any, reference: Any, candidate: Any
) -> tuple[bool, dict[str, Any]]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return False, {
            "finite": False,
            "normalized_l2_error": "shape_mismatch",
            "normalized_l2_limit": 1e-3,
            "max_abs_error": "shape_mismatch",
            "elementwise_tolerance": "0.015625 * max(abs(reference), 1)",
            "elementwise_tolerance_pass": False,
            "normalized_l2_pass": False,
            "bitwise_equal_fraction": 0.0,
            "shape_reference": list(reference.shape),
            "shape_candidate": list(candidate.shape),
        }
    finite = bool(torch.isfinite(reference).all().item()) and bool(
        torch.isfinite(candidate).all().item()
    )
    diff = (candidate.float() - reference.float()).abs()
    ref_float = reference.float()
    ref_norm = torch.linalg.vector_norm(ref_float)
    normalized_l2 = float(
        (torch.linalg.vector_norm(diff) / ref_norm.clamp_min(1e-12)).item()
    )
    max_abs = float(diff.max().item())
    bitwise_equal_fraction = float(
        (candidate.view(torch.int16) == reference.view(torch.int16))
        .float()
        .mean()
        .item()
    )
    allowed = 0.015625 * torch.maximum(
        ref_float.abs(), torch.ones_like(ref_float)
    )
    tolerance_mask = diff <= allowed
    tolerance_pass = bool(tolerance_mask.all().item())
    l2_pass = normalized_l2 <= 1e-3
    passed = finite and tolerance_pass and l2_pass
    details: dict[str, Any] = {
        "finite": finite,
        "normalized_l2_error": normalized_l2,
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


def _capture_graphs(
    torch: Any,
    device: Any,
    implementations: dict[str, Callable[[], Any]],
    flush_bytes: int,
    prime_reuse: bool,
) -> dict[str, dict[str, Any]]:
    if flush_bytes < 0:
        raise ValueError("flush_bytes must be non-negative")
    with torch.no_grad():
        for implementation in implementations.values():
            for _ in range(3):
                output = implementation()
                del output
        torch.cuda.synchronize(device)

        bundles: dict[str, dict[str, Any]] = {}
        for name, implementation in implementations.items():
            flush = (
                torch.empty(
                    flush_bytes // 4, device=device, dtype=torch.float32
                )
                if flush_bytes > 0
                else None
            )
            graph = torch.cuda.CUDAGraph()
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            prime_output = None
            try:
                with torch.cuda.graph(graph):
                    if flush is not None:
                        flush.zero_()
                    if prime_reuse:
                        prime_output = implementation()
                    start.record()
                    output = implementation()
                    end.record()
            except Exception as exc:
                raise RuntimeError(
                    "CUDA graph capture with external timing events failed for "
                    f"{name}: {exc}"
                ) from exc
            bundles[name] = {
                "graph": graph,
                "flush": flush,
                "start": start,
                "end": end,
                "output": output,
                "prime_output": prime_output,
            }
        torch.cuda.synchronize(device)
    return bundles


def _timing_case(
    torch: Any,
    device: Any,
    m: int,
    n: int,
    k: int,
    seed: int,
    implementations: dict[str, Callable[[], Any]],
    receipts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for regime, flush_bytes, prime_reuse in REGIMES:
        bundles = _capture_graphs(
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
                offset = (batch_index + preheat_batches) % len(names)
                order = names[offset:] + names[:offset]
                for name in order:
                    bundles[name]["graph"].replay()
                    preheat_replays += 1
            torch.cuda.synchronize(device)
            preheat_batches += 1
            if time.perf_counter() - preheat_started >= GRAPH_PREHEAT_SECONDS:
                break
        preheat_elapsed = time.perf_counter() - preheat_started
        if preheat_elapsed < GRAPH_PREHEAT_SECONDS:
            raise RuntimeError(
                f"graph preheat ended below {GRAPH_PREHEAT_SECONDS}s: {preheat_elapsed}"
            )

        samples: dict[str, list[float]] = {name: [] for name in names}
        for sample_index in range(TIMING_SAMPLES):
            offset = sample_index % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                bundle = bundles[name]
                bundle["graph"].replay()
                torch.cuda.synchronize(device)
                elapsed_ms = float(
                    bundle["start"].elapsed_time(bundle["end"])
                )
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise RuntimeError(
                        f"invalid CUDA event time for {regime}/{name}: {elapsed_ms}"
                    )
                samples[name].append(elapsed_ms)

        for name, values in samples.items():
            if len(values) != TIMING_SAMPLES:
                raise RuntimeError(
                    f"expected {TIMING_SAMPLES} samples for {regime}/{name}, got {len(values)}"
                )
            if name == REFERENCE_NAME:
                controls = {
                    "backend": "cutlass",
                    "cache_tactic": EXPECTED_CUTLASS_TACTIC,
                    "file_key": EXPECTED_CUTLASS_FILE_KEY,
                }
            else:
                controls = {
                    "load_path": receipts[name]["load_path"],
                    "use_prefetch": receipts[name]["use_prefetch"],
                    "tile": receipts[name]["tile"],
                    "cluster": receipts[name]["cluster"],
                    "enable_pdl": receipts[name]["enable_pdl"],
                    "swap_ab": receipts[name]["swap_ab"],
                }
            records.append(
                {
                    "record_type": "timing",
                    "m": m,
                    "n": n,
                    "k": k,
                    "seed": seed,
                    "implementation": name,
                    "controls": controls,
                    "regime": regime,
                    "median_ms": float(statistics.median(values)),
                    "raw_samples_ms": values,
                    "samples_ms": values,
                    "sample_count": len(values),
                    "preheat_wall_seconds": preheat_elapsed,
                    "preheat_batches": preheat_batches,
                    "preheat_replays": preheat_replays,
                    "output": _tensor_receipt(bundles[name]["output"]),
                    "compile_key": (
                        receipts[name]["compile_key"]
                        if name != REFERENCE_NAME
                        else None
                    ),
                    "graph_protocol": {
                        "external_events": True,
                        "flush_bytes": flush_bytes,
                        "prime_reuse": prime_reuse,
                        "timed_call_between_events": True,
                        "order_rotation": (
                            "preheat batch rotation and sample_index modulo "
                            "implementation count"
                        ),
                        "reuse_caveat": REUSE_CAVEAT,
                    },
                    "claim_scope": (
                        "synthetic CUDA-graph timing only; no runtime inference, "
                        "model quality, or end-to-end claim"
                    ),
                }
            )
        del bundles
    return records


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    output = _validate_output(args)
    server_cache = _validate_server_cache(args)
    handle = output.open("x", encoding="utf-8")
    context: dict[str, Any] = {"stage": "guards"}
    timing_records = 0
    try:
        _guard_environment()
        context = {"stage": "cache_snapshot"}
        clone_data, snapshot_data, cache_receipt = _read_cache_pair(server_cache)

        context = {"stage": "imports"}
        import cutlass
        import flashinfer
        import flashinfer.autotuner.autotuner as autotuner_module
        import flashinfer.fp4_quantization as flashinfer_compat_quantization
        import flashinfer.gemm.gemm_base as flashinfer_gemm_base
        import flashinfer.gemm.gemm_mm_fp4_cute_dsl as cute_dsl_module
        import flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x as dense_kernel_module
        import flashinfer.quantization.fp4_quantization as flashinfer_quantization
        import torch
        import torch.cuda.graphs as torch_graphs
        import torch.cuda.streams as torch_streams

        device, properties, free_bytes = _device_gate(torch)
        source_modules = {
            "flashinfer": flashinfer,
            "flashinfer.autotuner": autotuner_module,
            "flashinfer.gemm.gemm_base": flashinfer_gemm_base,
            "flashinfer.gemm.gemm_mm_fp4_cute_dsl": cute_dsl_module,
            "flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x": dense_kernel_module,
            "flashinfer.fp4_quantization_compat": flashinfer_compat_quantization,
            "flashinfer.quantization.fp4_quantization": flashinfer_quantization,
            "torch.cuda.graphs": torch_graphs,
            "torch.cuda.streams": torch_streams,
        }
        source_files = {
            name: _hash_file(_module_file(module))
            for name, module in source_modules.items()
        }
        profile_hash = _hash_file(PROFILE_SOURCE)
        source_files[EXPERIMENT] = profile_hash
        source_files["profile"] = profile_hash

        _emit(
            handle,
            {
                "record_type": "metadata",
                "experiment": EXPERIMENT,
                "status": "STARTED",
                "profile": {
                    "source": str(PROFILE_SOURCE),
                    "sha256": profile_hash["sha256"],
                },
                "shape": {"m": M, "k": K, "n": N},
                "shapes": [{"m": m, "k": k, "n": n} for m, k, n in SHAPES],
                "seeds": list(SEEDS),
                "candidates": [REFERENCE_NAME, *DIRECT_NAMES],
                "direct_contract": {
                    "kernel_a": "aq",
                    "kernel_b": "bq.T",
                    "sf_m": "(m + 127) // 128",
                    "sf_n": "(n + 127) // 128",
                    "sf_k": "(k // 16 + 3) // 4",
                    "logical_ab_dtype": str(cutlass.Float4E2M1FN),
                    "compiled_ab_dtype": str(cutlass.Uint8),
                    "sf_dtype": str(cutlass.Float8E4M3FN),
                    "c_dtype": str(cutlass.BFloat16),
                    "ab_assumed_align": 32,
                    "cluster_shape_mn": (1, 1),
                    "swap_ab": False,
                    "batch_size": 1,
                    "enable_pdl": True,
                    "tile": (64, 128),
                    "split_k_slices": 1,
                    "split_k_atomic_bf16": False,
                    "custom_epilogue": False,
                },
                "cache": {
                    **cache_receipt,
                    "expected_cutlass_tactic": EXPECTED_CUTLASS_TACTIC,
                    "expected_file_key": EXPECTED_CUTLASS_FILE_KEY,
                },
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
                },
                "source_files": source_files,
                "numerical_gate": {
                    "comparisons": len(DIRECT_NAMES) * len(SEEDS),
                    "normalized_l2_max": 1e-3,
                    "elementwise": "absdiff <= 0.015625 * max(abs(reference), 1)",
                    "finite_required": True,
                    "bitwise_fraction_reported": True,
                    "timing_requires_all_comparisons": True,
                },
                "timing_protocol": {
                    "implementations": 5,
                    "regimes": [
                        {"name": name, "flush_bytes": flush, "prime_reuse": prime}
                        for name, flush, prime in REGIMES
                    ],
                    "external_events": True,
                    "preheat_seconds_minimum": GRAPH_PREHEAT_SECONDS,
                    "samples_per_implementation": TIMING_SAMPLES,
                    "interleaved_with_rotating_order": True,
                    "expected_timing_records": 30,
                },
                "claim_scope": (
                    "synthetic numerical and CUDA-graph microbenchmark only; no "
                    "runtime inference, model quality, or end-to-end claim"
                ),
            },
        )

        context = {"stage": "cache_load"}
        tuner = autotuner_module.AutoTuner.get()
        if not tuner.load_configs(str(server_cache)):
            raise CacheMatchFailure("server cache metadata rejected")
        direct_kernel_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        device_index = int(torch.cuda.current_device())
        validated: list[
            tuple[
                int,
                int,
                int,
                int,
                dict[str, Callable[[], Any]],
                dict[str, dict[str, Any]],
            ]
        ] = []
        numerical_failures: list[dict[str, Any]] = []

        with autotuner_module.autotune(False):
            for seed in SEEDS:
                context = {"stage": "numerical", "m": M, "k": K, "n": N, "seed": seed}
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
                ) = _quantize_case(torch, flashinfer, device, seed)
                implementations, receipts = _make_implementations(
                    torch,
                    flashinfer,
                    cutlass,
                    device,
                    M,
                    N,
                    K,
                    aq,
                    bq,
                    asf,
                    bsf,
                    alpha,
                    dense_kernel_module.Sm120B12xBlockScaledDenseGemmKernel,
                    cute_dsl_module._compile_block_scaled_gemm,
                    cute_dsl_module._prepare_alpha_for_launch,
                    direct_kernel_cache,
                    device_index,
                    source_files,
                )
                for name in DIRECT_NAMES:
                    _emit(
                        handle,
                        {
                            "record_type": "direct_admission",
                            "m": M,
                            "k": K,
                            "n": N,
                            "seed": seed,
                            **receipts[name],
                        },
                    )
                _emit(
                    handle,
                    {
                        "record_type": "operands",
                        "m": M,
                        "k": K,
                        "n": N,
                        "seed": seed,
                        "a": _tensor_receipt(a),
                        "b": _tensor_receipt(b),
                        "a_packed": _tensor_receipt(aq),
                        "b_packed": _tensor_receipt(bq),
                        "b_packed_raw": _tensor_receipt(bq.T),
                        "kernel_a": _tensor_receipt(aq),
                        "kernel_b": _tensor_receipt(bq.T),
                        "kernel_a_sf": _tensor_receipt(asf),
                        "kernel_b_sf": _tensor_receipt(bsf.T),
                        "alpha_public_scalar": _tensor_receipt(alpha.reshape(())),
                        "a_global_scale": a_global,
                        "b_global_scale": b_global,
                    },
                )

                reference_before, initial_observations = _audit_search_cache(
                    tuner,
                    implementations[REFERENCE_NAME],
                    autotuner_module,
                )
                _check_output(torch, reference_before, M, N)
                initial_match = _require_cutlass_match(
                    initial_observations,
                    clone_data,
                    snapshot_data,
                    cache_receipt,
                    "before_direct_candidates",
                )
                _emit(
                    handle,
                    {
                        "record_type": "cache_match",
                        "m": M,
                        "k": K,
                        "n": N,
                        "seed": seed,
                        "output": _tensor_receipt(reference_before),
                        **initial_match,
                    },
                )

                direct_outputs: dict[str, Any] = {}
                with torch.no_grad():
                    for name in DIRECT_NAMES:
                        direct_outputs[name] = implementations[name]()
                        _emit(
                            handle,
                            {
                                "record_type": "direct_compile",
                                "m": M,
                                "k": K,
                                "n": N,
                                "seed": seed,
                                "implementation": name,
                                "output": _tensor_receipt(direct_outputs[name]),
                                **receipts[name],
                            },
                        )
                    reference_after, final_observations = _audit_search_cache(
                        tuner,
                        implementations[REFERENCE_NAME],
                        autotuner_module,
                    )
                torch.cuda.synchronize(device)
                _check_output(torch, reference_after, M, N)
                final_match = _require_cutlass_match(
                    final_observations,
                    clone_data,
                    snapshot_data,
                    cache_receipt,
                    "after_direct_candidates",
                )
                _emit(
                    handle,
                    {
                        "record_type": "cache_match",
                        "m": M,
                        "k": K,
                        "n": N,
                        "seed": seed,
                        "output": _tensor_receipt(reference_after),
                        **final_match,
                    },
                )
                _emit(
                    handle,
                    {
                        "record_type": "reference_output",
                        "m": M,
                        "k": K,
                        "n": N,
                        "seed": seed,
                        "phase": "after_direct_candidates",
                        "implementation": REFERENCE_NAME,
                        "output": _tensor_receipt(reference_after),
                        "source_hashes": source_files,
                    },
                )

                case_failed = False
                for name in DIRECT_NAMES:
                    candidate = direct_outputs[name]
                    passed, details = _failure_details(
                        torch, reference_after, candidate
                    )
                    record = {
                        "record_type": "numerical",
                        "m": M,
                        "k": K,
                        "n": N,
                        "seed": seed,
                        "implementation": name,
                        "reference": REFERENCE_NAME,
                        "load_path": receipts[name]["load_path"],
                        "use_prefetch": receipts[name]["use_prefetch"],
                        "can_implement": receipts[name]["can_implement"],
                        "compile_key": receipts[name]["compile_key"],
                        "output": _tensor_receipt(candidate),
                        "source_hashes": receipts[name]["source_hashes"],
                        "passed": passed,
                        **details,
                    }
                    _emit(handle, record)
                    if not passed:
                        case_failed = True
                        numerical_failures.append(record)

                del direct_outputs, reference_before, reference_after
                del a, b, aq, bq, asf, bsf, alpha
                if case_failed:
                    raise NumericalGateFailure(
                        "numerical gate failed: "
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
                validated.append((M, K, N, seed, implementations, receipts))

            if numerical_failures:
                raise NumericalGateFailure(
                    f"{len(numerical_failures)} numerical comparisons failed"
                )
            if len(validated) != len(SEEDS):
                raise NumericalGateFailure(
                    f"validated {len(validated)} seeds, expected {len(SEEDS)}"
                )

            _emit(
                handle,
                {
                    "record_type": "numerical_gate",
                    "status": "PASS",
                    "comparisons": len(DIRECT_NAMES) * len(SEEDS),
                    "timings_reported_only_after_all_comparisons": True,
                },
            )
            for m, k, n, seed, implementations, receipts in validated:
                context = {
                    "stage": "timing",
                    "m": m,
                    "k": k,
                    "n": n,
                    "seed": seed,
                }
                timing_case_records = _timing_case(
                    torch,
                    device,
                    m,
                    n,
                    k,
                    seed,
                    implementations,
                    receipts,
                )
                for record in timing_case_records:
                    _emit(handle, record)
                    timing_records += 1

        if timing_records != 30:
            raise RuntimeError(
                f"expected 30 timing records after the numerical gate, got {timing_records}"
            )
        _emit(
            handle,
            {
                "record_type": "summary",
                "experiment": EXPERIMENT,
                "status": "PASS",
                "final_status": "PASS",
                "timing_records": timing_records,
                "expected_timing_records": 30,
                "claim_scope": (
                    "synthetic numerical and CUDA-graph timing only; no runtime "
                    "inference, model quality, or end-to-end claim"
                ),
            },
        )
        return 0
    except NumericalGateFailure as exc:
        _emit(
            handle,
            {
                "record_type": "summary",
                "experiment": EXPERIMENT,
                "status": "FAIL",
                "final_status": "FAIL",
                "stage": context,
                "reason": str(exc),
                "timings_reported": False,
                "claim_scope": "no timing, inference, quality, or promotion claim",
            },
        )
        return 2
    except Exception as exc:
        _emit(
            handle,
            {
                "record_type": "error",
                "experiment": EXPERIMENT,
                "status": "ERROR",
                "final_status": "ERROR",
                "stage": context,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timings_reported": timing_records > 0,
                "claim_scope": "no runtime inference or model quality claim",
            },
        )
        return 1
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
