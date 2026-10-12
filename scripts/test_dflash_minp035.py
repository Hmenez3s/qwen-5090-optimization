from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dflash_minp035 import install, wrap

RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/minp035")
TMP_ROOT = Path("/tmp/qwen-minp035")
MIN_FREE_BYTES = 27000 * 1024 * 1024


class NumericalGateFailure(RuntimeError):
    pass


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DFlash target-only min-p probability oracle.")
    parser.add_argument("--output", required=True, help=f"new JSONL under {RUN_ROOT}/")
    return parser.parse_args(argv)


def _output(args: argparse.Namespace) -> Path:
    path = Path(args.output).expanduser()
    if not path.is_absolute() or path.suffix != ".jsonl":
        raise RuntimeError("--output must be an absolute .jsonl path")
    path = path.resolve()
    if not _under(path, RUN_ROOT) or path == RUN_ROOT or not path.parent.is_dir():
        raise RuntimeError(f"--output must be new and under {RUN_ROOT}")
    if path.exists():
        raise RuntimeError(f"refusing existing output: {path}")
    return path


def _guards() -> None:
    if not _under(Path.cwd().resolve(), RUN_ROOT):
        raise RuntimeError(f"cwd must be inside {RUN_ROOT}")
    roots = {
        "HOME": (RUN_ROOT,), "TMPDIR": (TMP_ROOT,),
        "CUDA_CACHE_PATH": (RUN_ROOT, TMP_ROOT),
        "CUTE_DSL_CACHE_DIR": (RUN_ROOT, TMP_ROOT),
        "CUTE_DSL_DUMP_DIR": (RUN_ROOT, TMP_ROOT),
        "FLASHINFER_WORKSPACE_BASE": (RUN_ROOT,), "SGLANG_CACHE_DIR": (RUN_ROOT,),
        "TRITON_CACHE_DIR": (RUN_ROOT,), "TORCHINDUCTOR_CACHE_DIR": (RUN_ROOT,),
        "TORCH_EXTENSIONS_DIR": (RUN_ROOT,),
    }
    for name, allowed in roots.items():
        raw = os.environ.get(name)
        if not raw:
            raise RuntimeError(f"{name} must be set; no fallback is permitted")
        value = Path(raw).expanduser()
        if not value.is_absolute() or not any(_under(value.resolve(), root) for root in allowed):
            raise RuntimeError(f"{name} is outside its owned root")
        if not value.is_dir():
            raise RuntimeError(f"{name} must name an existing directory")


def _device(torch: Any) -> tuple[Any, Any, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device0 is unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if (int(props.major), int(props.minor)) != (12, 0):
        raise RuntimeError(f"device0 must be SM120, got SM{props.major}{props.minor}")
    free, _ = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"device0 needs 27000 MiB free, got {free}")
    return device, props, int(free)


def _hash(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


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


def _zero_fastpath() -> dict[str, Any]:
    class Info:
        need_min_p_sampling = False

    calls = 0
    sentinel = object()

    def spy(**kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        return sentinel

    wrapped = wrap(spy)
    observed = wrapped(sampling_info=Info(), bs=1, draft_token_num=2)
    if observed is not sentinel or calls != 1:
        raise NumericalGateFailure(f"zero fast path mismatch: calls={calls}, same={observed is sentinel}")
    if str(inspect.signature(wrapped)) != str(inspect.signature(spy)):
        raise NumericalGateFailure("functools.wraps did not preserve signature")
    return {"record_type": "zero_fastpath", "status": "PASS", "calls": calls, "same_object": True}


def _info(torch: Any, device: Any, min_ps: list[float]) -> Any:
    return SimpleNamespace(
        temperatures=torch.ones((2, 1), device=device, dtype=torch.float32),
        top_ps=torch.tensor([1.0, 0.75], device=device, dtype=torch.float32),
        top_ks=torch.tensor([4, 3], device=device, dtype=torch.int32),
        min_ps=torch.tensor(min_ps, device=device, dtype=torch.float32),
        need_top_k_sampling=True, need_top_p_sampling=True, need_min_p_sampling=True,
    )


def _kwargs(torch: Any, device: Any, info: Any, sparse: bool) -> dict[str, Any]:
    # The fifth token is -inf, making max_top_k=4 < vocab=5 genuinely sparse.
    logprobs = [[.55, .30, .10, .05, 0.0]] * 3 + [[.40, .30, .20, .10, 0.0]] * 3
    next_token_logits = torch.tensor(logprobs, device=device).log()
    return dict(next_token_logits=next_token_logits, sampling_info=info, draft_token_num=3,
                bs=2, max_top_k=4, uniform_top_k_value=None, use_sparse_topk=sparse)


def _reference(torch: Any, raw: Any, min_ps: Any) -> Any:
    minimum = min_ps.to(device=raw.device, dtype=raw.dtype).view(-1, 1, 1)
    filtered = raw.masked_fill(raw < raw.amax(dim=-1, keepdim=True) * minimum, 0)
    normalized = filtered / filtered.sum(dim=-1, keepdim=True)
    return torch.where(minimum == 0, raw, normalized)


def _installed_cases(torch: Any, device: Any, original: Any, corrected: Any,
                     dflash_utils: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    original_topk = dflash_utils._dflash_top_k_renorm_prob
    topk_calls = 0

    def probe(*args: Any, **kwargs: Any) -> Any:
        nonlocal topk_calls
        topk_calls += 1
        return original_topk(*args, **kwargs)

    dflash_utils._dflash_top_k_renorm_prob = probe
    try:
        for sparse in (False, True):
            info = _info(torch, device, [0.5, 0.0])
            kwargs = _kwargs(torch, device, info, sparse)
            calls_before = topk_calls
            raw = original(**kwargs)
            actual = corrected(**kwargs)
            branch_calls = topk_calls - calls_before
            expected_branch_calls = 0 if sparse else 2  # raw + corrected dense builds
            if branch_calls != expected_branch_calls:
                raise NumericalGateFailure(f"top-k branch probe mismatch: sparse={sparse}, calls={branch_calls}")
            expected = _reference(torch, raw, info.min_ps)
            if not bool(torch.allclose(actual, expected, rtol=2e-6, atol=2e-6)):
                raise NumericalGateFailure(json.dumps({"case": "reference", "sparse": sparse,
                    "raw": raw.detach().cpu().tolist(), "observed": actual.detach().cpu().tolist(),
                    "expected": expected.detach().cpu().tolist()}))
            if not bool(torch.equal(raw[1], actual[1])):
                raise NumericalGateFailure(f"min-p-zero row changed in {'sparse' if sparse else 'dense'} case")
            if not bool(((raw[0] > 0) & (actual[0] == 0)).any()):
                raise NumericalGateFailure(f"positive min-p removed no mass in {'sparse' if sparse else 'dense'} case")
            if not bool(torch.allclose(actual.sum(-1), torch.ones_like(actual.sum(-1)), rtol=2e-6, atol=2e-6)):
                raise NumericalGateFailure(f"rows do not sum to one in {'sparse' if sparse else 'dense'} case")
            records.append({"record_type": "installed_reference", "status": "PASS", "sparse": sparse,
                            "max_top_k": 4, "vocab": int(raw.shape[-1]), "topk_probe_calls": branch_calls,
                            "raw_has_excluded_tokens": bool((raw == 0).any()),
                            "raw": raw.detach().cpu().tolist(), "corrected": actual.detach().cpu().tolist()})

            zero_info = _info(torch, device, [0.0, 0.0])
            zero_kwargs = _kwargs(torch, device, zero_info, sparse)
            zero_raw = original(**zero_kwargs)
            zero_actual = corrected(**zero_kwargs)
            if not bool(torch.equal(zero_raw, zero_actual)):
                raise NumericalGateFailure(f"all-zero min-p branch changed in {'sparse' if sparse else 'dense'} case")
            records.append({"record_type": "zero_branch", "status": "PASS", "sparse": sparse,
                            "max_top_k": 4, "vocab": int(zero_raw.shape[-1]), "same_tensor_values": True})
        return records
    finally:
        dflash_utils._dflash_top_k_renorm_prob = original_topk


def _direct(torch: Any, device: Any) -> list[dict[str, Any]]:
    def identity(**kwargs: Any) -> Any:
        return kwargs["next_token_logits"]

    q = torch.tensor([[[.5, .25, .15, .10], [.4, .3, .2, .1]],
                      [[.5, .25, .15, .10], [.4, .3, .2, .1]]], dtype=torch.float64, device=device)
    q_before = q.clone()
    info = SimpleNamespace(min_ps=torch.tensor([.5, 0.0], dtype=q.dtype, device=device), need_min_p_sampling=True)
    actual = wrap(identity)(next_token_logits=q, sampling_info=info, bs=2, draft_token_num=2)
    expected = torch.tensor([2 / 3, 1 / 3, 0, 0], dtype=q.dtype, device=device)
    expected_second = torch.tensor([4 / 9, 1 / 3, 2 / 9, 0], dtype=q.dtype, device=device)
    if (not bool(torch.allclose(actual[0, 0], expected, rtol=1e-12, atol=1e-12))
            or not bool(torch.allclose(actual[0, 1], expected_second, rtol=1e-12, atol=1e-12))
            or not bool(torch.equal(actual[1], q[1])) or not bool(torch.equal(q, q_before))):
        raise NumericalGateFailure(json.dumps({"case": "direct", "observed": actual.tolist(), "input": q.tolist()}))
    tie_q = torch.tensor([[[.4, .4, .2, 0.0], [.25, .25, .25, .25]]], dtype=torch.float64, device=device)
    tie_info = SimpleNamespace(min_ps=torch.ones(1, dtype=q.dtype, device=device), need_min_p_sampling=True)
    ties = wrap(identity)(next_token_logits=tie_q, sampling_info=tie_info, bs=1, draft_token_num=2)
    if not bool(torch.equal(ties[0, 0], torch.tensor([.5, .5, 0, 0], dtype=q.dtype, device=device))) or not bool(torch.equal(ties[0, 1], tie_q[0, 1])):
        raise NumericalGateFailure(json.dumps({"case": "min_p_one_ties", "observed": ties.tolist()}))
    return [{"record_type": "direct", "status": "PASS", "expected_first_row": [2 / 3, 1 / 3, 0, 0],
             "zero_rows_bitwise_unchanged": True, "min_p_one_ties_retained": True}]


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    output = _output(args)
    handle = output.open("x", encoding="utf-8")
    context: dict[str, Any] = {"stage": "guards"}
    try:
        _guards()
        _emit(handle, _zero_fastpath())
        context = {"stage": "imports"}
        import torch
        from sglang.srt.speculative import dflash_utils
        device, props, free = _device(torch)
        preinstalled = dflash_utils.build_dflash_verify_target_probs
        if bool(getattr(preinstalled, "_dflash_minp035_installed", False)):
            original = getattr(preinstalled, "_dflash_minp035_original", None)
            if original is None:
                raise NumericalGateFailure("installed marker lacks its original builder")
            dflash_utils.build_dflash_verify_target_probs = original
        else:
            original = preinstalled
        corrected = install()
        if (dflash_utils.build_dflash_verify_target_probs is not corrected or install() is not corrected
                or str(inspect.signature(corrected)) != str(inspect.signature(original))):
            raise NumericalGateFailure("installation was not globally replaced once/idempotently")
        source = Path(inspect.getmodule(original).__file__).resolve()
        _emit(handle, {"record_type": "metadata", "experiment": "dflash_minp035", "status": "STARTED",
            "script": _hash(Path(__file__).resolve()), "helper": _hash(Path(__file__).with_name("dflash_minp035.py")),
            "installed_builder_source": _hash(source), "preinstalled_before_test": bool(getattr(preinstalled, "_dflash_minp035_installed", False)),
            "device": {"name": str(props.name), "major": int(props.major),
            "minor": int(props.minor), "free_memory": free}, "versions": {"python": sys.version, "torch": str(torch.__version__)},
            "claim_scope": "synthetic target-only probability construction; no model-quality claim"})
        context = {"stage": "direct"}
        for record in _direct(torch, device):
            _emit(handle, record)
        context = {"stage": "installed_builder"}
        for record in _installed_cases(torch, device, original, corrected, dflash_utils):
            _emit(handle, record)
        _emit(handle, {"record_type": "summary", "experiment": "dflash_minp035", "status": "PASS"})
        return 0
    except NumericalGateFailure as exc:
        _emit(handle, {"record_type": "summary", "experiment": "dflash_minp035", "status": "FAIL", "stage": context, "error": str(exc)})
        return 2
    except Exception as exc:
        _emit(handle, {"record_type": "error", "experiment": "dflash_minp035", "status": "ERROR", "stage": context,
                       "error_type": type(exc).__name__, "error": str(exc)})
        return 1
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
