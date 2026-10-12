"""Fail-closed private exp038 hook: owned AOT plus min-p wrapper observability."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
from typing import Any


OWN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/exp038-minp")
HELPER_SHA256 = "eb43e977b6f2174819c4bc53c7f475b90f0bc5448fea6ab7e73754b4e7b4f2cb"
DFLASH_UTILS_SHA256 = "fb72a3a28d5ecb33d77c6cd6655cc5b4abc8086feba5fb879c44b98b5cbb8374"
_HELPER_NAME = "dflash_minp035.py"
_AOT_NAME = "sitecustomize-prebuilt032.py"
_OBSERVABILITY_MARKER = "_dflash_minp038_observability_installed"


def _fail(message: str) -> None:
    raise SystemExit("exp038 sitecustomize failed: " + message)


def _owned_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        _fail(f"owned file is not a physical regular file: {path}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_marker(prefix: str, record: dict[str, Any]) -> None:
    print(prefix + " " + json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


def _load_owned_helper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_dflash_minp035_exp038", path)
    if spec is None or spec.loader is None:
        _fail("could not load the owned min-p helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observed_wrapper(corrected: Any) -> Any:
    if bool(getattr(corrected, _OBSERVABILITY_MARKER, False)):
        return corrected

    @functools.wraps(corrected)
    def observed(*args: Any, **kwargs: Any) -> Any:
        result = corrected(*args, **kwargs)
        info = kwargs.get("sampling_info")
        if bool(getattr(info, "need_min_p_sampling", False)):
            # Tensor values are never inspected. shape is tensor metadata and does
            # not trigger a device read or synchronization.
            shape = [int(dimension) for dimension in getattr(result, "shape", ())]
            _json_marker(
                "DFLASH_MINP038_POSITIVE",
                {
                    "pid": os.getpid(),
                    "bs": int(kwargs["bs"]),
                    "draft_tokens": int(kwargs["draft_token_num"]),
                    "shape": shape,
                },
            )
        return result

    setattr(observed, _OBSERVABILITY_MARKER, True)
    setattr(observed, "_dflash_minp038_original", corrected)
    return observed


def _install() -> None:
    root = Path(__file__).resolve().parent
    if root != OWN_ROOT or root.is_symlink() or root.resolve() != root:
        _fail(f"hook must execute from its own physical root {OWN_ROOT}")

    helper_path = _owned_file(root, _HELPER_NAME)
    aot_path = _owned_file(root, _AOT_NAME)
    helper_hash = _sha256(helper_path)
    if helper_hash != HELPER_SHA256:
        _fail(f"helper hash mismatch: {helper_hash}")

    # Keep the established AOT implementation single-sourced.
    runpy.run_path(str(aot_path), run_name="__sitecustomize_prebuilt032__")

    from sglang.srt.speculative import dflash_utils

    module_path = Path(dflash_utils.__file__).resolve()
    if not module_path.is_file() or module_path.is_symlink():
        _fail(f"dflash_utils source is not a physical file: {module_path}")
    dflash_hash = _sha256(module_path)
    if dflash_hash != DFLASH_UTILS_SHA256:
        _fail(f"dflash_utils hash mismatch: {dflash_hash}")

    helper = _load_owned_helper(helper_path)
    current = dflash_utils.build_dflash_verify_target_probs
    if not bool(getattr(current, _OBSERVABILITY_MARKER, False)):
        corrected = helper.install()
        dflash_utils.build_dflash_verify_target_probs = _observed_wrapper(corrected)

    _json_marker(
        "DFLASH_MINP038_STARTUP",
        {
            "pid": os.getpid(),
            "root": str(root),
            "helper_sha256": helper_hash,
            "dflash_utils_sha256": dflash_hash,
            "wrapper": _OBSERVABILITY_MARKER,
        },
    )


try:
    _install()
except SystemExit:
    raise
except BaseException as exc:
    # sitecustomize commonly suppresses ordinary exceptions; this hook is fail-closed.
    raise SystemExit("exp038 sitecustomize failed: " + str(exc)) from exc
