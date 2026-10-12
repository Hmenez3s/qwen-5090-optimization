"""Offline-safe client for measuring native SGLang or OpenAI Chat streaming.

Usage: benchmark_stream.py --url http://127.0.0.1:PORT --expected-model MODEL
  --manifest MANIFEST.json --output RESULTS.jsonl [--protocol native|openai]
  [--timeout SECONDS]

The client accepts only caller-supplied requests, adds only protocol-required
stream fields, opens results exclusively, performs model identity preflight,
never retries,
and records every SSE data event. Decode excludes the first positive-token
chunk; unavailable or zero-duration rates are null rather than fabricated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


URL_RE = re.compile(r"^http://127\.0\.0\.1:([0-9]+)$")
PROTOCOLS = frozenset({"native", "openai"})
ALLOWED_PAYLOAD_KEYS = frozenset({"input_ids", "text", "sampling_params", "return_logprob"})
OPENAI_PAYLOAD_KEYS = frozenset({"messages", "temperature", "top_p", "max_tokens"})
OPENAI_ROLES = frozenset({"system", "user", "assistant"})
OPENAI_MAX_TEMPERATURE = 2.0
OPENAI_MAX_TOP_P = 1.0
OPENAI_MAX_TOKENS = 131072
REQUIRED_USAGE_FIELDS = (
    "completion_tokens",
    "prompt_tokens",
    "cached_tokens",
    "finish_reason",
)
OPTIONAL_SPEC_FIELDS = ("spec_accept_length", "spec_verify_ct")


class ValidationError(ValueError):
    """The local manifest or CLI value is outside the benchmark contract."""


class BenchmarkError(RuntimeError):
    """A request or native streaming protocol failure."""


class StreamProtocolError(BenchmarkError):
    def __init__(
        self,
        message: str,
        *,
        events: list[dict[str, Any]] | None = None,
        final_text: Any = None,
        final_meta: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.events = events or []
        self.final_text = final_text
        self.final_meta = dict(final_meta) if final_meta is not None else None


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect rather than allowing endpoint escape."""

    def _reject(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any) -> Any:
        raise urllib.error.HTTPError(req.full_url, code, "redirects disabled", headers, fp)

    http_error_301 = _reject
    http_error_302 = _reject
    http_error_303 = _reject
    http_error_307 = _reject
    http_error_308 = _reject


def build_isolated_opener() -> urllib.request.OpenerDirector:
    """Build an opener with no proxy fallback and no redirects."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirectHandler(),
    )


def validate_protocol(protocol: Any) -> str:
    if not isinstance(protocol, str) or protocol not in PROTOCOLS:
        raise ValidationError("protocol must be native or openai")
    return protocol


def validate_url(url: str) -> str:
    if not isinstance(url, str) or not URL_RE.fullmatch(url):
        raise ValidationError("--url must be exactly http://127.0.0.1:PORT")
    port = int(URL_RE.fullmatch(url).group(1))
    if not 1 <= port <= 65535:
        raise ValidationError("--url port must be between 1 and 65535")
    if port in (8081, 8002):
        raise ValidationError("--url ports 8081 and 8002 are forbidden")
    return url


def validate_timeout(timeout: Any) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValidationError("--timeout must be a finite positive number")
    value = float(timeout)
    if not math.isfinite(value) or value <= 0 or value > 180:
        raise ValidationError("--timeout must be > 0 and <= 180 seconds")
    return value


def _reject_remote_or_forbidden_fields(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(
                marker in lowered
                for marker in (
                    "session",
                    "load",
                    "path",
                    "adapter",
                    "multimodal",
                    "remote",
                    "url",
                    "image",
                    "audio",
                    "video",
                    "remote_url",
                    "image_url",
                    "audio_url",
                    "video_url",
                )
            ):
                raise ValidationError(f"forbidden remote/session field: {path}.{key}")
            _reject_remote_or_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_remote_or_forbidden_fields(child, f"{path}[{index}]")


def _finite_bounded_number(value: Any, *, name: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number between {minimum} and {maximum}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
        raise ValidationError(f"{name} must be a finite number between {minimum} and {maximum}")


def _validate_native_payload(payload: dict[str, Any], case_name: str) -> dict[str, Any]:
    unknown = set(payload) - ALLOWED_PAYLOAD_KEYS
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValidationError(f"{case_name}: unsupported payload field(s): {names}")
    if "stream" in payload:
        raise ValidationError(f"{case_name}: stream is controlled by the client")
    has_ids = "input_ids" in payload
    has_text = "text" in payload
    if has_ids == has_text:
        raise ValidationError(f"{case_name}: provide exactly one of input_ids or text")
    if has_ids:
        input_ids = payload["input_ids"]
        if not isinstance(input_ids, list) or not input_ids:
            raise ValidationError(f"{case_name}: input_ids must be a non-empty list")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in input_ids):
            raise ValidationError(f"{case_name}: input_ids must contain non-negative integers")
    elif not isinstance(payload["text"], str):
        raise ValidationError(f"{case_name}: text must be a string")
    if "sampling_params" not in payload or not isinstance(payload["sampling_params"], dict):
        raise ValidationError(f"{case_name}: sampling_params must be an object")
    if "return_logprob" in payload and not isinstance(payload["return_logprob"], bool):
        raise ValidationError(f"{case_name}: return_logprob must be boolean")
    _reject_remote_or_forbidden_fields(payload["sampling_params"], f"{case_name}.sampling_params")
    return copy.deepcopy(payload)


def _validate_openai_payload(payload: dict[str, Any], case_name: str) -> dict[str, Any]:
    unknown = set(payload) - OPENAI_PAYLOAD_KEYS
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValidationError(f"{case_name}: unsupported OpenAI payload field(s): {names}")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValidationError(f"{case_name}: messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValidationError(f"{case_name}: message {index} must contain only role and content")
        if not isinstance(message["role"], str) or message["role"] not in OPENAI_ROLES:
            raise ValidationError(f"{case_name}: message {index} has an unsupported role")
        if not isinstance(message["content"], str):
            raise ValidationError(f"{case_name}: message {index} content must be a plain string")
    if "temperature" in payload:
        _finite_bounded_number(payload["temperature"], name=f"{case_name}.temperature", minimum=0.0, maximum=OPENAI_MAX_TEMPERATURE)
    if "top_p" in payload:
        _finite_bounded_number(payload["top_p"], name=f"{case_name}.top_p", minimum=1.0e-12, maximum=OPENAI_MAX_TOP_P)
    if "max_tokens" not in payload:
        raise ValidationError(f"{case_name}: max_tokens is required")
    max_tokens = payload["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 0 < max_tokens <= OPENAI_MAX_TOKENS:
        raise ValidationError(f"{case_name}.max_tokens must be a positive integer <= {OPENAI_MAX_TOKENS}")
    return copy.deepcopy(payload)


def validate_payload(payload: Any, case_name: str = "case", protocol: str = "native") -> dict[str, Any]:
    validate_protocol(protocol)
    if not isinstance(payload, dict):
        raise ValidationError(f"{case_name}: payload must be an object")
    return _validate_native_payload(payload, case_name) if protocol == "native" else _validate_openai_payload(payload, case_name)


def validate_manifest(manifest: Any, protocol: str = "native") -> dict[str, Any]:
    validate_protocol(protocol)
    if not isinstance(manifest, dict) or set(manifest) != {"cases", "repetitions"}:
        raise ValidationError("manifest must contain exactly cases and repetitions")
    repetitions = manifest["repetitions"]
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValidationError("manifest repetitions must be a positive integer")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValidationError("manifest cases must be a non-empty list")
    seen: set[str] = set()
    normalized_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"name", "payload"}:
            raise ValidationError(f"case {index} must contain exactly name and payload")
        name = case["name"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValidationError(f"case {index} name must be a unique non-empty string")
        seen.add(name)
        normalized_cases.append({"name": name, "payload": validate_payload(case["payload"], name, protocol)})
    return {"cases": normalized_cases, "repetitions": repetitions}


def load_manifest(path: str | os.PathLike[str], protocol: str = "native") -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValidationError("manifest path must be an existing regular file")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationError(f"invalid manifest JSON: {exc}") from exc
    return validate_manifest(manifest, protocol)


def _open_output_exclusive(path: str | os.PathLike[str]):
    output_path = Path(path)
    if not output_path.name:
        raise ValidationError("output must be a new regular-file path")
    if not output_path.parent.is_dir():
        raise ValidationError("output parent directory must already exist")
    try:
        descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ValidationError("output path already exists; refusing overwrite") from exc
    except OSError as exc:
        raise ValidationError(f"cannot create output exclusively: {exc}") from exc
    return output_path, os.fdopen(descriptor, "w", encoding="utf-8")


def _write_record(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    handle.flush()


def _status(response: Any) -> int | None:
    value = getattr(response, "status", None)
    if value is None and hasattr(response, "getcode"):
        value = response.getcode()
    return int(value) if value is not None else None


def _body_text(response: Any) -> str:
    body = response.read()
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def _request(opener: Any, request: urllib.request.Request, timeout: float) -> Any:
    response = opener.open(request, timeout=timeout)
    status = _status(response)
    if status is not None and not 200 <= status < 300:
        try:
            detail = _body_text(response)
        finally:
            response.close()
        raise BenchmarkError(f"HTTP status {status}: {detail}")
    return response


def _models(opener: Any, base_url: str, expected_model: str, timeout: float) -> list[str]:
    request = urllib.request.Request(
        base_url + "/v1/models",
        method="GET",
        headers={"Accept": "application/json"},
    )
    try:
        response = _request(opener, request, timeout)
        try:
            decoded = json.loads(_body_text(response))
        finally:
            response.close()
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TypeError) as exc:
        raise BenchmarkError(f"model preflight failed: {exc}") from exc
    if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), list):
        raise BenchmarkError("model preflight response must contain a data list")
    model_ids = [entry.get("id") for entry in decoded["data"] if isinstance(entry, dict) and isinstance(entry.get("id"), str)]
    if expected_model not in model_ids:
        raise BenchmarkError(f"expected model identity not served: {expected_model}")
    return model_ids


def _request_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _declared_settings(payload: Mapping[str, Any], protocol: str) -> dict[str, Any]:
    if protocol == "openai":
        return copy.deepcopy(dict(payload))
    settings: dict[str, Any] = {
        "input_kind": "input_ids" if "input_ids" in payload else "text",
        "sampling_params": copy.deepcopy(payload["sampling_params"]),
    }
    if "return_logprob" in payload:
        settings["return_logprob"] = payload["return_logprob"]
    return settings


def _protocol_error(message: str, events: list[dict[str, Any]], final_data: dict[str, Any] | None = None) -> StreamProtocolError:
    return StreamProtocolError(
        message,
        events=events,
        final_text=final_data.get("text") if final_data else None,
        final_meta=final_data.get("meta_info") if final_data and isinstance(final_data.get("meta_info"), dict) else None,
    )


def _measure_native_stream(
    response: Any,
    request_started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Consume one native SSE response and return evidence plus safe metrics."""
    events: list[dict[str, Any]] = []
    counts: list[tuple[float, int]] = []
    final_data: dict[str, Any] | None = None
    previous_time = request_started
    try:
        for raw_line in response:
            now = clock()
            if not math.isfinite(now) or not math.isfinite(request_started) or now < request_started or now < previous_time:
                raise _protocol_error("non-finite or non-monotonic event timing", events, final_data)
            previous_time = now
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            line = line.rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data_text = line[5:].lstrip(" ")
            elapsed = now - request_started
            if data_text == "":
                events.append({"raw_sse": line, "seconds": elapsed, "completion_tokens": None})
                continue
            if data_text == "[DONE]":
                events.append({"raw_sse": line, "seconds": elapsed, "done": True, "completion_tokens": None})
                continue
            try:
                data = json.loads(data_text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise _protocol_error(f"malformed SSE JSON: {exc}", events, final_data) from exc
            if not isinstance(data, dict):
                raise _protocol_error("SSE data must be a JSON object", events, final_data)
            if data.get("error") is not None:
                raise _protocol_error("error SSE event received", events, final_data)
            meta = data.get("meta_info")
            if meta is not None and not isinstance(meta, dict):
                raise _protocol_error("meta_info must be an object", events, data)
            count: int | None = None
            if isinstance(meta, dict) and "completion_tokens" in meta:
                candidate = meta["completion_tokens"]
                if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 0:
                    raise _protocol_error("completion_tokens must be a non-negative integer", events, data)
                count = candidate
                if counts and candidate < counts[-1][1]:
                    raise _protocol_error("completion_tokens are not monotonic", events, data)
                counts.append((elapsed, candidate))
            event = {"raw_sse": line, "seconds": elapsed, "completion_tokens": count}
            if "text" in data:
                event["text"] = data["text"]
            events.append(event)
            final_data = data
    except StreamProtocolError:
        raise
    except (UnicodeDecodeError, OSError, ValueError) as exc:
        raise _protocol_error(f"SSE stream failed: {exc}", events, final_data) from exc
    finally:
        response.close()

    if final_data is None:
        raise _protocol_error("SSE contained no JSON data event", events)
    final_meta = final_data.get("meta_info")
    if not isinstance(final_meta, dict):
        raise _protocol_error("final SSE event is missing meta_info", events, final_data)
    finish = final_meta.get("finish_reason")
    if isinstance(finish, dict) and finish.get("type") == "abort":
        raise _protocol_error("server aborted request: " + str(finish.get("message", "unspecified")), events, final_data)
    missing = [field for field in REQUIRED_USAGE_FIELDS if field not in final_meta]
    if missing:
        raise _protocol_error("final SSE event missing usage fields: " + ", ".join(missing), events, final_data)
    completion = final_meta["completion_tokens"]
    if isinstance(completion, bool) or not isinstance(completion, int) or completion < 0:
        raise _protocol_error("final completion_tokens must be a non-negative integer", events, final_data)
    if not counts or counts[-1][1] != completion:
        raise _protocol_error("final completion_tokens missing from monotonic SSE counts", events, final_data)
    for field in ("prompt_tokens", "cached_tokens"):
        value = final_meta[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise _protocol_error(f"final {field} must be finite and non-negative", events, final_data)
    for field in OPTIONAL_SPEC_FIELDS:
        value = final_meta.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0):
            raise _protocol_error(f"final {field} must be finite and non-negative when present", events, final_data)
    finish_reason = final_meta["finish_reason"]
    if isinstance(finish_reason, str) and finish_reason:
        finish_reason_type = finish_reason
    elif isinstance(finish_reason, dict) and isinstance(finish_reason.get("type"), str) and finish_reason["type"]:
        finish_reason_type = finish_reason["type"]
    else:
        raise _protocol_error("final finish_reason must be a non-empty string or typed object", events, final_data)
    ended = clock()
    if not math.isfinite(ended) or ended < request_started or ended < previous_time:
        raise _protocol_error("non-finite or non-monotonic completion timing", events, final_data)
    total_seconds = ended - request_started
    first_time, first_count = next(((t, c) for t, c in counts if c > 0), (None, None))
    increasing = [
        (timestamp, count)
        for index, (timestamp, count) in enumerate(counts)
        if count > 0 and (index == 0 or count > counts[index - 1][1])
    ]
    last_time, last_count = increasing[-1] if increasing else (None, None)
    decode_seconds = None if first_time is None or last_time is None else last_time - first_time
    decode_tps = None
    if first_time is not None and last_time is not None and decode_seconds and decode_seconds > 0 and last_count > first_count:
        decode_tps = (last_count - first_count) / decode_seconds
    e2e_tps = completion / total_seconds if completion > 0 and total_seconds > 0 else None
    return {
        "status": "ok",
        "events": events,
        "text": final_data.get("text"),
        "meta": copy.deepcopy(final_meta),
        "finish_reason": finish_reason,
        "finish_reason_type": finish_reason_type,
        "completion_tokens": completion,
        "metrics": {
            "ttft_seconds": first_time,
            "decode_tokens_per_second": decode_tps,
            "e2e_tokens_per_second": e2e_tps,
            "total_seconds": total_seconds,
        },
    }


def _measure_openai_stream(
    response: Any,
    request_started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Consume OpenAI Chat SSE, separating visible content and reasoning deltas."""
    events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    first_nonempty: float | None = None
    last_nonempty: float | None = None
    final_data: dict[str, Any] | None = None
    final_usage: dict[str, Any] | None = None
    finish_reason: Any = None
    previous_time = request_started
    try:
        for raw_line in response:
            now = clock()
            if not math.isfinite(now) or not math.isfinite(request_started) or now < request_started or now < previous_time:
                raise _protocol_error("non-finite or non-monotonic event timing", events, final_data)
            previous_time = now
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            line = line.rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data_text = line[5:].lstrip(" ")
            elapsed = now - request_started
            if data_text == "":
                events.append({"raw_sse": line, "seconds": elapsed})
                continue
            if data_text == "[DONE]":
                events.append({"raw_sse": line, "seconds": elapsed, "done": True})
                continue
            try:
                data = json.loads(data_text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise _protocol_error(f"malformed SSE JSON: {exc}", events, final_data) from exc
            if not isinstance(data, dict):
                raise _protocol_error("SSE data must be a JSON object", events, final_data)
            if data.get("error") is not None:
                raise _protocol_error("error SSE event received", events, data)
            choices = data.get("choices")
            if not isinstance(choices, list):
                raise _protocol_error("OpenAI SSE data must contain a choices list", events, data)
            event: dict[str, Any] = {"raw_sse": line, "seconds": elapsed}
            if choices:
                if len(choices) != 1 or not isinstance(choices[0], dict):
                    raise _protocol_error("OpenAI SSE choices must contain one object", events, data)
                choice = choices[0]
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    raise _protocol_error("OpenAI SSE choice delta must be an object", events, data)
                # SGLang emits explicit nulls in role-only Chat deltas.
                content = "" if delta.get("content") is None else delta["content"]
                reasoning = "" if delta.get("reasoning_content") is None else delta["reasoning_content"]
                for label, value in (("content", content), ("reasoning_content", reasoning)):
                    if not isinstance(value, str):
                        raise _protocol_error(f"OpenAI delta.{label} must be a string", events, data)
                    if value:
                        if first_nonempty is None:
                            first_nonempty = elapsed
                        last_nonempty = elapsed
                        event[label] = value
                        (content_parts if label == "content" else reasoning_parts).append(value)
                if "finish_reason" in choice and choice["finish_reason"] is not None:
                    finish_reason = choice["finish_reason"]
                    if finish_reason == "abort":
                        raise _protocol_error("server aborted request", events, data)
                    event["finish_reason"] = finish_reason
            usage = data.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise _protocol_error("OpenAI usage must be an object or null", events, data)
                completion = usage.get("completion_tokens")
                if isinstance(completion, bool) or not isinstance(completion, int) or completion < 0:
                    raise _protocol_error("OpenAI completion_tokens must be a non-negative integer", events, data)
                final_usage = copy.deepcopy(usage)
                event["usage"] = copy.deepcopy(usage)
            events.append(event)
            final_data = data
    except StreamProtocolError:
        raise
    except (UnicodeDecodeError, OSError, ValueError) as exc:
        raise _protocol_error(f"SSE stream failed: {exc}", events, final_data) from exc
    finally:
        response.close()

    if final_data is None:
        raise _protocol_error("SSE contained no JSON data event", events)
    if not isinstance(final_data.get("usage"), dict) or final_usage is None or "completion_tokens" not in final_usage:
        raise _protocol_error("final OpenAI SSE usage is missing completion_tokens", events, final_data)
    if not isinstance(finish_reason, str) or not finish_reason:
        raise _protocol_error("final OpenAI finish_reason must be a non-empty string", events, final_data)
    ended = clock()
    if not math.isfinite(ended) or ended < request_started or ended < previous_time:
        raise _protocol_error("non-finite or non-monotonic completion timing", events, final_data)
    total_seconds = ended - request_started
    completion = final_usage["completion_tokens"]
    stream_seconds = None if first_nonempty is None or last_nonempty is None else last_nonempty - first_nonempty
    stream_estimate = completion / stream_seconds if stream_seconds is not None and stream_seconds > 0 else None
    e2e_tps = completion / total_seconds if completion > 0 and total_seconds > 0 else None
    return {
        "status": "ok",
        "events": events,
        "text": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "reasoning_content": "".join(reasoning_parts),
        "usage": copy.deepcopy(final_usage),
        "finish_reason": finish_reason,
        "finish_reason_type": finish_reason,
        "completion_tokens": completion,
        "metrics": {
            "ttft_seconds": first_nonempty,
            "decode_tokens_per_second": None,
            "stream_tokens_per_second_estimate": stream_estimate,
            "e2e_tokens_per_second": e2e_tps,
            "total_seconds": total_seconds,
        },
        "metric_notes": [
            "OpenAI chunks have no per-chunk token counter; decode_tokens_per_second is null.",
            "stream_tokens_per_second_estimate uses completion_tokens over the first-to-last nonempty content/reasoning delta window and may have first-chunk and hidden-token bias.",
            "Usage-only final delay is included in total_seconds but excluded from the last nonempty delta timestamp.",
        ],
    }


def measure_stream(
    response: Any,
    request_started: float,
    clock: Callable[[], float] = time.perf_counter,
    protocol: str = "native",
) -> dict[str, Any]:
    """Consume one explicitly selected SSE protocol."""
    validate_protocol(protocol)
    if protocol == "native":
        return _measure_native_stream(response, request_started, clock)
    return _measure_openai_stream(response, request_started, clock)


def run_benchmark(
    *,
    url: str,
    expected_model: str,
    manifest: Mapping[str, Any],
    output: str | os.PathLike[str],
    timeout: float = 180,
    opener: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
    protocol: str = "native",
) -> None:
    """Run validated repetitions; errors are recorded and stop later cases."""
    selected_protocol = validate_protocol(protocol)
    base_url = validate_url(url)
    if not isinstance(expected_model, str) or not expected_model:
        raise ValidationError("expected model must be a non-empty string")
    normalized = validate_manifest(manifest, selected_protocol)
    bounded_timeout = validate_timeout(timeout)
    _, handle = _open_output_exclusive(output)
    active_opener = opener if opener is not None else build_isolated_opener()
    try:
        try:
            model_ids = _models(active_opener, base_url, expected_model, bounded_timeout)
            _write_record(handle, {"record_type": "preflight", "status": "ok", "protocol": selected_protocol, "expected_model": expected_model, "served_models": model_ids})
        except Exception as exc:
            _write_record(handle, {"record_type": "error", "phase": "model_preflight", "error": str(exc), "expected_model": expected_model})
            raise
        for case in normalized["cases"]:
            declared = case["payload"]
            request_payload = copy.deepcopy(declared)
            if selected_protocol == "openai":
                request_payload.update({
                    "model": expected_model,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                })
                endpoint = "/v1/chat/completions"
            else:
                request_payload["stream"] = True
                endpoint = "/generate"
            request_hash = _request_hash(request_payload)
            for repetition in range(1, normalized["repetitions"] + 1):
                request_started = clock()
                record: dict[str, Any] = {
                    "record_type": "case",
                    "status": "error",
                    "case": case["name"],
                    "repetition": repetition,
                    "expected_model": expected_model,
                    "protocol": selected_protocol,
                    "request_sha256": request_hash,
                    "declared_settings": _declared_settings(declared, selected_protocol),
                }
                response = None
                try:
                    body = json.dumps(request_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
                    request = urllib.request.Request(
                        base_url + endpoint,
                        data=body,
                        method="POST",
                        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
                    )
                    response = _request(active_opener, request, bounded_timeout)
                    measurement = measure_stream(response, request_started, clock, selected_protocol)
                    record.update(measurement)
                except StreamProtocolError as exc:
                    record.update({
                        "events": exc.events,
                        "text": exc.final_text,
                        "meta": exc.final_meta,
                        "error": str(exc),
                    })
                except Exception as exc:
                    if response is not None:
                        response.close()
                    record["error"] = str(exc) or exc.__class__.__name__
                _write_record(handle, record)
                if record["status"] != "ok":
                    raise BenchmarkError(record["error"])
    finally:
        handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure native SGLang or OpenAI Chat streaming without proxy or redirect fallback")
    parser.add_argument("--url", required=True)
    parser.add_argument("--protocol", choices=("native", "openai"), default="native")
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=180)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest, args.protocol)
        run_benchmark(
            url=args.url,
            expected_model=args.expected_model,
            manifest=manifest,
            output=args.output,
            timeout=args.timeout,
            protocol=args.protocol,
        )
    except (ValidationError, BenchmarkError, OSError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
