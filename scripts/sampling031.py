from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/sampling031")
TMP_ROOT = Path("/tmp/qwen-sampling031")
MIN_FREE_BYTES = 27000 * 1024 * 1024
B = 65536


class NumericalGateFailure(RuntimeError):
    pass


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone target-only speculative-sampling oracle.")
    parser.add_argument("--output", required=True, help=f"new JSONL under {RUN_ROOT}/")
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


def _guard_environment(run_root: Path = RUN_ROOT, tmp_root: Path = TMP_ROOT) -> None:
    cwd = Path.cwd().resolve()
    if not _under(cwd, run_root):
        raise RuntimeError(f"cwd must be inside {run_root}, got {cwd}")
    env_roots = {
        "HOME": (run_root,),
        "TMPDIR": (tmp_root,),
        "CUDA_CACHE_PATH": (run_root, tmp_root),
        "CUTE_DSL_CACHE_DIR": (run_root, tmp_root),
        "CUTE_DSL_DUMP_DIR": (run_root, tmp_root),
        "FLASHINFER_WORKSPACE_BASE": (run_root,),
        "SGLANG_CACHE_DIR": (run_root,),
        "TRITON_CACHE_DIR": (run_root,),
        "TORCHINDUCTOR_CACHE_DIR": (run_root,),
        "TORCH_EXTENSIONS_DIR": (run_root,),
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
            raise RuntimeError(f"{name}={value} is outside allowed roots")
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


def _device_gate(torch: Any) -> tuple[Any, Any, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device0 is unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if (int(properties.major), int(properties.minor)) != (12, 0):
        raise RuntimeError(f"device0 must be SM120, got SM{properties.major}{properties.minor}")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"device0 needs at least {MIN_FREE_BYTES} free bytes, got {free_bytes}")
    return device, properties, int(free_bytes)


def _loaded_libraries(torch: Any) -> dict[str, Any]:
    found: dict[tuple[str, str], None] = {}
    try:
        loaded = getattr(torch.ops, "loaded_libraries", None)
        for value in loaded or ():
            path = str(value)
            if "/sgl_kernel/" in path or "sgl_kernel" in Path(path).name.lower() or "sgl-kernel" in Path(path).name.lower():
                found[("torch.ops.loaded_libraries", path)] = None
    except Exception:
        pass
    for name, module in list(sys.modules.items()):
        value = getattr(module, "__file__", None)
        if value and (name == "sgl_kernel" or name.startswith("sgl_kernel.") or "/sgl_kernel/" in str(value)):
            if Path(value).suffix.lower() in {".so", ".dylib", ".dll"}:
                found[(f"sys.modules:{name}", str(Path(value).resolve()))] = None
    entries = [{"source": source, "path": path} for source, path in sorted(found)]
    for entry in entries:
        path = Path(entry["path"])
        if path.is_file():
            entry.update({"sha256": _hash_file(path)["sha256"], "bytes": path.stat().st_size})
        else:
            entry.update({"sha256": "unavailable", "reason": "visible path is not a regular file"})
    return {"status": "observed" if entries else "unavailable", "libraries": entries}


def _make_inputs(torch: Any, device: Any, q0: list[float], q1: list[float], proposal: int,
                 uniforms: Any = None, final_uniform: Any = None, generator: Any = None) -> dict[str, Any]:
    vocab = len(q0)
    target_probs = torch.empty((B if generator is not None else 1, 2, vocab), device=device, dtype=torch.float32)
    batch = target_probs.shape[0]
    target_probs[:, 0, :] = torch.tensor(q0, device=device, dtype=torch.float32)
    target_probs[:, 1, :] = torch.tensor(q1, device=device, dtype=torch.float32)
    candidates = torch.zeros((batch, 2), device=device, dtype=torch.int64)
    candidates[:, 1] = proposal
    retrive_index = torch.arange(batch * 2, device=device, dtype=torch.int64).reshape(batch, 2)
    retrive_next_token = torch.tensor([1, -1], device=device, dtype=torch.int64).repeat(batch, 1)
    retrive_next_sibling = torch.tensor([-1, -1], device=device, dtype=torch.int64).repeat(batch, 1)
    if generator is None:
        uniform_samples = torch.tensor(uniforms, device=device, dtype=torch.float32)
        uniform_samples_for_final_sampling = torch.tensor(final_uniform, device=device, dtype=torch.float32)
    else:
        uniform_samples = torch.rand((batch, 2), device=device, dtype=torch.float32, generator=generator)
        uniform_samples_for_final_sampling = torch.rand((batch,), device=device, dtype=torch.float32, generator=generator)
    return {
        "predicts": torch.full((batch * 2,), -1, device=device, dtype=torch.int32),
        "accept_index": torch.full((batch, 2), -1, device=device, dtype=torch.int32),
        "accept_token_num": torch.zeros((batch,), device=device, dtype=torch.int32),
        "candidates": candidates,
        "retrive_index": retrive_index,
        "retrive_next_token": retrive_next_token,
        "retrive_next_sibling": retrive_next_sibling,
        "uniform_samples": uniform_samples,
        "uniform_samples_for_final_sampling": uniform_samples_for_final_sampling,
        "target_probs": target_probs,
        "draft_probs": torch.zeros_like(target_probs),
    }


def _invoke(torch: Any, op: Any, device: Any, tensors: dict[str, Any], threshold_single: float = 1.0,
            threshold_acc: float = 1.0) -> dict[str, Any]:
    args = [tensors[name] for name in ("predicts", "accept_index", "accept_token_num", "candidates",
        "retrive_index", "retrive_next_token", "retrive_next_sibling", "uniform_samples",
        "uniform_samples_for_final_sampling", "target_probs", "draft_probs")]
    op(*args, threshold_single, threshold_acc, True)
    torch.cuda.synchronize(device)
    return {name: value.detach().cpu() for name, value in tensors.items()}


def _check_rows(result: dict[str, Any], vocab: int) -> tuple[Any, Any]:
    predicts, retrieve = result["predicts"], result["retrive_index"]
    first = predicts[retrieve[:, 0]]
    accepted = result["accept_token_num"]
    indices = result["accept_index"]
    if not bool(((first >= 0) & (first < vocab)).all()):
        raise NumericalGateFailure("first-token output escaped vocabulary")
    if not bool(((accepted == 0) | (accepted == 1)).all()):
        raise NumericalGateFailure("accept_token_num escaped {0,1}")
    if not bool((indices[:, 0] == retrieve[:, 0]).all()):
        raise NumericalGateFailure("accept_index root is invalid")
    second_ok = ((accepted == 1) & (indices[:, 1] == retrieve[:, 1])) | ((accepted == 0) & (indices[:, 1] == -1))
    if not bool(second_ok.all()):
        raise NumericalGateFailure("accept_index second entry is invalid")
    return first, accepted


def _deterministic(torch: Any, op: Any, device: Any) -> list[dict[str, Any]]:
    records = []
    q0, q1 = [0.6, 0.4], [0.5, 0.5]
    accepted = _invoke(torch, op, device, _make_inputs(torch, device, q0, q1, 0, [[0.5, 0.0]], [0.25]))
    first, nums = _check_rows(accepted, 2)
    if first.tolist() != [0] or accepted["predicts"].tolist() != [0, 0] or nums.tolist() != [1] or accepted["accept_index"].tolist() != [[0, 1]]:
        raise NumericalGateFailure("deterministic acceptance fixture mismatch")
    records.append({"record_type": "deterministic_fixture", "name": "accept", "status": "PASS", "first": first.tolist(), "accept_token_num": nums.tolist(), "accept_index": accepted["accept_index"].tolist()})
    rejected = _invoke(torch, op, device, _make_inputs(torch, device, q0, q1, 0, [[0.8, 0.0]], [0.25]))
    first, nums = _check_rows(rejected, 2)
    draft = rejected["draft_probs"][0, 0].tolist()
    if first.tolist() != [1] or nums.tolist() != [0] or rejected["accept_index"].tolist() != [[0, -1]] or abs(float(draft[0]) - 0.6) > 1e-6 or draft[1] != 0.0:
        raise NumericalGateFailure("deterministic rejection fixture mismatch")
    records.append({"record_type": "deterministic_fixture", "name": "reject", "status": "PASS", "first": first.tolist(), "accept_token_num": nums.tolist(), "draft_probs_root": draft})
    return records


def _statistical(torch: Any, op: Any, device: Any, seed: int, q0: list[float], q1: list[float], proposal: int) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    result = _invoke(torch, op, device, _make_inputs(torch, device, q0, q1, proposal, generator=generator))
    first, accepted = _check_rows(result, len(q0))
    counts = torch.bincount(first, minlength=len(q0)).tolist()
    frequencies = [count / B for count in counts]
    tolerances = [6 * math.sqrt(p * (1 - p) / B) + 2 / B for p in q0]
    errors = [abs(f - p) for f, p in zip(frequencies, q0)]
    accept_rate = float((accepted == 1).float().mean().item())
    accept_expected = q0[proposal]
    accept_tolerance = 6 * math.sqrt(accept_expected * (1 - accept_expected) / B) + 2 / B
    if any(error > tolerance for error, tolerance in zip(errors, tolerances)) or abs(accept_rate - accept_expected) > accept_tolerance:
        raise NumericalGateFailure(json.dumps({'fixture': 'statistical', 'seed': seed, 'proposal': proposal, 'counts': counts, 'expected': q0, 'tolerances': tolerances, 'accept_rate': accept_rate, 'accept_expected': accept_expected}))
    return {"record_type": "statistical_fixture", "status": "PASS", "seed": seed, "proposal": proposal,
            "counts": counts, "frequencies": frequencies, "expected": q0, "tolerance": tolerances,
            "maxerror": max(errors), "accept_count": int((accepted == 1).sum().item()),
            "accept_rate": accept_rate, "accept_expected": accept_expected, "accept_tolerance": accept_tolerance,
            "accept_maxerror": abs(accept_rate - accept_expected)}


def _negative_control(torch: Any, op: Any, device: Any) -> dict[str, Any]:
    q0, q1 = [0.2, 0.8], [0.5, 0.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    result = _invoke(torch, op, device, _make_inputs(torch, device, q0, q1, 0, generator=generator), 1.0, 0.5)
    first, _ = _check_rows(result, 2)
    counts = torch.bincount(first, minlength=2).tolist()
    frequencies = [count / B for count in counts]
    biased = [0.4, 0.6]
    tolerance = [6 * math.sqrt(p * (1 - p) / B) + 2 / B for p in biased]
    errors = [abs(f - p) for f, p in zip(frequencies, biased)]
    deviation = abs(frequencies[0] - q0[0])
    if any(error > limit for error, limit in zip(errors, tolerance)) or deviation <= 0.1:
        raise NumericalGateFailure("negative-control bias fixture failed")
    return {"record_type": "negative_control", "status": "PASS", "label": "expected bias; never recommended server config",
            "threshold_acc": 0.5, "counts": counts, "frequencies": frequencies, "predicted_biased": biased,
            "tolerance": tolerance, "maxerror": max(errors), "original_q0": q0,
            "token0_deviation_from_original": deviation}


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli(argv)
    output = _validate_output(args)
    handle = output.open("x", encoding="utf-8")
    context: dict[str, Any] = {"stage": "guards"}
    try:
        _guard_environment()
        context = {"stage": "imports"}
        import torch
        import sgl_kernel.speculative as wrapper
        from sgl_kernel.speculative import tree_speculative_sampling_target_only
        device, properties, free_bytes = _device_gate(torch)
        package_version = importlib.metadata.version("sglang-kernel")
        if package_version != "0.4.6.post1":
            raise RuntimeError(f"expected sglang-kernel==0.4.6.post1, got {package_version}")
        metadata = {"record_type": "metadata", "experiment": "sampling031", "status": "STARTED",
            "package": {"name": "sglang-kernel", "version": package_version},
            "wrapper": _hash_file(Path(wrapper.__file__).resolve()),
            "script": _hash_file(Path(__file__).resolve()), "loaded_sgl_kernel_libraries": _loaded_libraries(torch),
            "device": {"name": str(properties.name), "major": int(properties.major), "minor": int(properties.minor),
                       "total_memory": int(properties.total_memory), "free_memory_before_allocations": free_bytes},
            "versions": {"python": sys.version, "torch": str(torch.__version__), "torch_cuda": str(torch.version.cuda)},
            "seeds": [17, 29, 43], "wrapper_call": {"style": "positional", "arity": 14},
            "threshold_reference": {"threshold_single": 1.0, "threshold_acc": 1.0},
            "source_vs_runtime_caveat": "Runtime metadata and installed wrapper/op only; upstream source is contextual evidence, not provenance proof.",
            "resource_protocol": "GPU execution is parent-owned; this oracle performs no model/server/network/cache cleanup operation."}
        _emit(handle, metadata)
        context = {"stage": "deterministic"}
        for record in _deterministic(torch, tree_speculative_sampling_target_only, device):
            _emit(handle, record)
        context = {"stage": "statistical"}
        for seed in (17, 29, 43):
            for q0, proposal in (([0.6, 0.25, 0.1, 0.05], 0), ([0.6, 0.25, 0.1, 0.05], 3), ([0.0, 0.2, 0.3, 0.5], 0)):
                context = {'stage': 'statistical', 'seed': seed, 'proposal': proposal, 'q0': q0}
                _emit(handle, _statistical(torch, tree_speculative_sampling_target_only, device, seed, q0, [0.1, 0.2, 0.3, 0.4], proposal))
        context = {"stage": "negative_control"}
        _emit(handle, _negative_control(torch, tree_speculative_sampling_target_only, device))
        _emit(handle, {"record_type": "summary", "experiment": "sampling031", "status": "PASS", "claim_scope": "synthetic target-only compiled-sampler oracle; no product behavior or server recommendation"})
        return 0
    except NumericalGateFailure as exc:
        _emit(handle, {"record_type": "summary", "experiment": "sampling031", "status": "FAIL", "stage": context, "error": str(exc)})
        return 2
    except Exception as exc:
        _emit(handle, {"record_type": "error", "experiment": "sampling031", "status": "ERROR", "stage": context, "error_type": type(exc).__name__, "error": str(exc)})
        return 1
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
