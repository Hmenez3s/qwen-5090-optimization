"""Target-only DFLASH min-p correction, kept import-safe for sitecustomize."""

from __future__ import annotations

import functools
import operator
from typing import Any, Callable


_MARKER = "_dflash_minp035_installed"


def _index(value: Any, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(result)


def wrap(original_builder: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the installed dense DFLASH probability builder with min-p filtering.

    The original builder supplies finite, normalized probabilities and upstream
    SamplingParams validates min_p in [0, 1]. Thus at least one maximum survives
    each positive-row filter; no clipping/fallback conceals invalid distributions.
    """
    if not callable(original_builder):
        raise TypeError("original_builder must be callable")

    @functools.wraps(original_builder)
    def corrected_builder(*args: Any, **kwargs: Any) -> Any:
        # The installed API is keyword-only; call it exactly once before inspecting
        # the optional correction path so the zero/false path is allocation-free.
        result = original_builder(*args, **kwargs)
        try:
            sampling_info = kwargs["sampling_info"]
        except KeyError as exc:
            raise TypeError("sampling_info is required by the DFLASH builder") from exc
        if not bool(getattr(sampling_info, "need_min_p_sampling", False)):
            return result

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - positive path needs torch
            raise RuntimeError("min-p correction requires torch") from exc

        if not isinstance(result, torch.Tensor) or result.ndim != 3:
            raise ValueError("DFLASH target probabilities must be a rank-3 Tensor")
        bs = _index(kwargs.get("bs"), "bs")
        draft_token_num = _index(kwargs.get("draft_token_num"), "draft_token_num")
        if tuple(result.shape[:2]) != (bs, draft_token_num):
            raise ValueError(
                "DFLASH target probability shape does not match bs/draft_token_num"
            )
        if not result.dtype.is_floating_point:
            raise TypeError("DFLASH target probabilities must have floating dtype")

        min_ps = getattr(sampling_info, "min_ps", None)
        if min_ps is None or not isinstance(min_ps, torch.Tensor):
            raise TypeError("sampling_info.min_ps must be a Tensor on min-p path")
        if min_ps.ndim != 1 or tuple(min_ps.shape) != (bs,):
            raise ValueError(f"sampling_info.min_ps must have shape [{bs}]")
        if not min_ps.dtype.is_floating_point:
            raise TypeError("sampling_info.min_ps must have floating dtype")

        min_ps = min_ps.to(device=result.device, dtype=result.dtype).view(bs, 1, 1)
        cutoff = result.amax(dim=-1, keepdim=True) * min_ps
        filtered = result.masked_fill(result < cutoff, 0)
        normalized = filtered / filtered.sum(dim=-1, keepdim=True)
        # Do not renormalize min-p==0 rows: where selects the original values
        # exactly, including mixed batches and the original zero support.
        return torch.where(min_ps == 0, result, normalized)

    return corrected_builder


def install() -> Callable[..., Any]:
    """Install this wrapper on dflash_utils and return the bound builder."""
    from sglang.srt.speculative import dflash_utils

    current = dflash_utils.build_dflash_verify_target_probs
    if bool(getattr(current, _MARKER, False)):
        return current
    wrapped = wrap(current)
    setattr(wrapped, _MARKER, True)
    setattr(wrapped, "_dflash_minp035_original", current)
    dflash_utils.build_dflash_verify_target_probs = wrapped
    return wrapped
