"""Final synthetic quality, reasoning, tools, stochastic min-p, and integration gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from benchmark_stream import build_isolated_opener


RUN_ROOT = Path("/home/operator/qwen38-5090/optimization-runs/pi-serving-053")
URL = "http://127.0.0.1:8081"
EXPECTED_MODEL = "qwen3.8-27b-normal"
LAUNCH = RUN_ROOT / "launch.json"
PID_RECEIPT = RUN_ROOT / "server-pid.json"
STDOUT_LOG = Path("/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/pi-serving-053/stdout.log")
OUTPUT = RUN_ROOT / "results/quality-final.jsonl"
HELPER_SHA256 = "eb43e977b6f2174819c4bc53c7f475b90f0bc5448fea6ab7e73754b4e7b4f2cb"
DFLASH_UTILS_SHA256 = "fb72a3a28d5ecb33d77c6cd6655cc5b4abc8086feba5fb879c44b98b5cbb8374"
KDA_SHA256 = "b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5"
TIMEOUT = 180


class QualityFailure(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Run the private synthetic exp038 min-p integration quality gate."
    )


def _physical(path: Path, *, directory: bool = False) -> None:
    if path.is_symlink() or path.resolve() != path:
        raise QualityFailure(f"path is not physical: {path}")
    if directory and not path.is_dir():
        raise QualityFailure(f"directory is missing: {path}")
    if not directory and not path.is_file():
        raise QualityFailure(f"file is missing: {path}")


def _json_file(path: Path) -> dict[str, Any]:
    _physical(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityFailure(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityFailure(f"JSON object required in {path}")
    return value


def _proc_snapshot(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text(encoding="utf-8").split()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise QualityFailure(f"process identity unavailable for {pid}: {exc}") from exc
    if len(fields) <= 21 or not boot_id:
        raise QualityFailure(f"invalid process identity for {pid}")
    try:
        parent = int(fields[3])
    except (IndexError, ValueError) as exc:
        raise QualityFailure(f"invalid process parent for {pid}") from exc
    return {
        "pid": pid,
        "start": fields[21],
        "ppid": parent,
        "state": fields[2],
        "comm": fields[1].strip("()"),
        "boot_id": boot_id,
    }


def _preflight() -> tuple[dict[str, Any], int]:
    _physical(RUN_ROOT, directory=True)
    launch = _json_file(LAUNCH)
    if launch.get("cwd") != str(RUN_ROOT) or launch.get("port") != 8081:
        raise QualityFailure("launch receipt is not owned by pi-serving-053 at port 8081")
    argv = launch.get("argv")
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise QualityFailure("launch argv is invalid")
    try:
        port_index = argv.index("--port")
        model_index = argv.index("--served-model-name")
    except ValueError as exc:
        raise QualityFailure("launch argv is missing required identity flags") from exc
    if (
        port_index + 1 >= len(argv)
        or argv[port_index + 1] != "8081"
        or model_index + 1 >= len(argv)
        or argv[model_index + 1] != EXPECTED_MODEL
    ):
        raise QualityFailure("launch argv port/model identity mismatch")

    receipt = _json_file(PID_RECEIPT)
    pid = receipt.get("pid")
    stored_stat = receipt.get("stat")
    boot_id = receipt.get("boot_id")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise QualityFailure("server PID receipt is invalid")
    if not isinstance(stored_stat, str) or len(stored_stat.split()) <= 21:
        raise QualityFailure("server PID receipt stat is invalid")
    if not isinstance(boot_id, str) or not boot_id:
        raise QualityFailure("server PID receipt boot_id is invalid")
    stored_start = stored_stat.split()[21]
    current = _proc_snapshot(pid)
    if current["boot_id"] != boot_id or current["start"] != stored_start or current["state"] == "Z":
        raise QualityFailure("server PID receipt no longer identifies the live process")
    main_identity = {"pid": pid, "start": stored_start, "boot_id": boot_id}

    _physical(STDOUT_LOG)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise QualityFailure(f"refusing existing output: {OUTPUT}")
    if not OUTPUT.parent.is_dir() or OUTPUT.parent.resolve() != OUTPUT.parent:
        raise QualityFailure("results directory is not an owned physical directory")
    log_bytes = STDOUT_LOG.read_bytes()
    _require_startup_marker(log_bytes.decode("utf-8", errors="replace"), main_identity)
    return main_identity, len(log_bytes)


def _marker(line: str, prefix: str) -> dict[str, Any] | None:
    if not line.startswith(prefix + " "):
        return None
    try:
        value = json.loads(line[len(prefix) + 1 :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _startup_pids(log: str) -> set[int]:
    pids: set[int] = set()
    for line in log.splitlines():
        value = _marker(line, "FINAL053_STARTUP")
        pid = value.get("pid") if value is not None else None
        if (
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and value.get("root") == str(RUN_ROOT)
            and value.get("helper_sha256") == HELPER_SHA256
            and value.get("dflash_utils_sha256") == DFLASH_UTILS_SHA256
            and value.get("minp_observer") == "_dflash_minp053_observer"
            and value.get("kda_sha256") == KDA_SHA256
            and value.get("kda_shapes") == [[8, 5120, 248320], [8, 17408, 5120]]
        ):
            pids.add(pid)
    return pids


def _require_startup_marker(log: str, main_identity: dict[str, Any]) -> None:
    if main_identity["pid"] not in _startup_pids(log):
        raise QualityFailure("owned startup hook identity marker for main server is absent")


def _write(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    handle.flush()


def _request(
    opener: Any,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    min_p: float,
    max_tokens: int,
    thinking: bool,
    seed: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": EXPECTED_MODEL,
        "messages": messages,
        "temperature": temperature,
        "min_p": min_p,
        "top_p": 1,
        "top_k": 50,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if seed is not None:
        body["seed"] = seed
    if tools is not None:
        body["tools"] = tools
    request = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            decoded = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise QualityFailure(f"HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TypeError) as exc:
        raise QualityFailure(f"synthetic request failed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise QualityFailure("chat response must be an object")
    return decoded


def _choice(response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise QualityFailure("chat response has no choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise QualityFailure("chat response choice has no message")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning, str):
        raise QualityFailure("reasoning field is not text")
    return choice, message, reasoning


def _objects(text: Any) -> list[Any]:
    found: list[Any] = []
    if not isinstance(text, str):
        return found
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            found.append(json.JSONDecoder().raw_decode(text[index:])[0])
        except json.JSONDecodeError:
            continue
    return found


def _arithmetic(
    handle: Any,
    opener: Any,
    name: str,
    prompt: str,
    expected: dict[str, Any],
) -> bool:
    try:
        response = _request(
            opener,
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            min_p=0.1,
            max_tokens=4096,
            thinking=True,
        )
        choice, message, reasoning = _choice(response)
        passed = expected in _objects(message.get("content")) and choice.get("finish_reason") == "stop"
        _write(handle, {
            "name": name,
            "passed": passed,
            "finish_reason": choice.get("finish_reason"),
            "reasoning": reasoning,
            "response": response,
        })
        if not passed:
            raise QualityFailure(f"arithmetic oracle failed: {name}")
        return bool(reasoning)
    except Exception as exc:
        if not isinstance(exc, QualityFailure) or "arithmetic oracle failed" not in str(exc):
            _write(handle, {"name": name, "passed": False, "error": str(exc)})
        raise


def _tool_definition() -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers and return their sum.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }]


def _tool_roundtrip(handle: Any, opener: Any) -> None:
    tools = _tool_definition()
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": "Use add_numbers to calculate 17+25. After receiving its result, return final JSON with key sum.",
    }]
    first = _request(opener, messages, temperature=0.7, min_p=0.1, max_tokens=4096, thinking=True, tools=tools)
    choice, message, reasoning = _choice(first)
    calls = message.get("tool_calls") or []
    emission_passed = (
        len(calls) == 1
        and isinstance(calls[0], dict)
        and calls[0].get("function", {}).get("name") == "add_numbers"
    )
    arguments: dict[str, Any] | None = None
    if emission_passed:
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            emission_passed = False
    emission_passed = emission_passed and arguments == {"a": 17, "b": 25}
    _write(handle, {
        "name": "tool-emission",
        "passed": emission_passed,
        "finish_reason": choice.get("finish_reason"),
        "reasoning": reasoning,
        "response": first,
    })
    if not emission_passed:
        raise QualityFailure("tool emission oracle failed")

    assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content"), "tool_calls": calls}
    for key in ("reasoning_content", "reasoning"):
        if message.get(key):
            assistant[key] = message[key]
    messages.extend([assistant, {
        "role": "tool",
        "tool_call_id": calls[0]["id"],
        "content": json.dumps({"result": 42}),
    }])
    final = _request(opener, messages, temperature=0.7, min_p=0.1, max_tokens=4096, thinking=True, tools=tools)
    final_choice, final_message, final_reasoning = _choice(final)
    passed = {"sum": 42} in _objects(final_message.get("content")) and final_choice.get("finish_reason") == "stop"
    _write(handle, {
        "name": "tool-roundtrip",
        "passed": passed,
        "finish_reason": final_choice.get("finish_reason"),
        "reasoning": final_reasoning,
        "response": final,
    })
    if not passed:
        raise QualityFailure("tool roundtrip oracle failed")


def _seed_cases(handle: Any, opener: Any) -> None:
    for seed in (17, 29, 43):
        name = f"seed-{seed}"
        try:
            response = _request(
                opener,
                [{"role": "user", "content": "Compute 987 times 123. Return only JSON with key product."}],
                temperature=0.7,
                min_p=0.1,
                max_tokens=4096,
                thinking=True,
                seed=seed,
            )
            choice, message, reasoning = _choice(response)
            passed = {"product": 121401} in _objects(message.get("content")) and choice.get("finish_reason") == "stop"
            _write(handle, {
                "name": name,
                "seed": seed,
                "passed": passed,
                "finish_reason": choice.get("finish_reason"),
                "reasoning": reasoning,
                "response": response,
            })
            if not passed:
                raise QualityFailure(f"seed oracle failed: {seed}")
        except Exception as exc:
            if not isinstance(exc, QualityFailure) or "seed oracle failed" not in str(exc):
                _write(handle, {"name": name, "seed": seed, "passed": False, "error": str(exc)})
            raise


def _optional_pair(handle: Any, opener: Any) -> None:
    prompt = "Compute 987 times 123. Return JSON with key product."
    records: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str]] = []
    for name, temperature, min_p in (("greedy", 0.0, 0.0), ("sampled", 0.7, 1.0)):
        response = _request(
            opener,
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            min_p=min_p,
            max_tokens=128,
            thinking=False,
        )
        choice, message, reasoning = _choice(response)
        passed = {"product": 121401} in _objects(message.get("content")) and choice.get("finish_reason") == "stop"
        _write(handle, {
            "name": f"optional-{name}",
            "passed": passed,
            "finish_reason": choice.get("finish_reason"),
            "reasoning": reasoning,
            "response": response,
        })
        if not passed:
            raise QualityFailure(f"optional semantic oracle failed: {name}")
        records.append((name, response, choice, message, reasoning))
    _write(handle, {
        "name": "optional-equality-observation",
        "passed": True,
        "equality_observed": records[0][3].get("content") == records[1][3].get("content"),
        "claim_scope": "response equality is an observation only; it does not prove a distribution",
    })


def _ancestry_reaches_main(pid: int, main_pid: int, boot_id: str) -> bool:
    seen: set[int] = set()
    current = pid
    for _ in range(64):
        if current == main_pid:
            return True
        if current in seen or current <= 1:
            return False
        seen.add(current)
        snapshot = _proc_snapshot(current)
        if snapshot["boot_id"] != boot_id or snapshot["state"] == "Z":
            return False
        if snapshot["comm"] == "resource_tracker":
            return False
        current = snapshot["ppid"]
    return False


def _require_post_test_marker(log_before: int, main_identity: dict[str, Any]) -> dict[str, Any]:
    data = STDOUT_LOG.read_bytes()
    if len(data) <= log_before:
        raise QualityFailure("server log did not append after requests")
    full_log = data.decode("utf-8", errors="replace")
    startup_pids = _startup_pids(full_log)
    suffix = data[log_before:].decode("utf-8", errors="replace")
    for line in suffix.splitlines():
        value = _marker(line, "DFLASH_MINP053_POSITIVE")
        marker_pid = value.get("pid") if value is not None else None
        if (
            not isinstance(marker_pid, int)
            or isinstance(marker_pid, bool)
            or marker_pid not in startup_pids
            or not isinstance(value.get("bs"), int)
            or isinstance(value.get("bs"), bool)
            or not isinstance(value.get("draft_tokens"), int)
            or isinstance(value.get("draft_tokens"), bool)
            or not isinstance(value.get("shape"), list)
            or len(value["shape"]) != 3
            or value.get("observer_removed") is not True
        ):
            continue
        main_now = _proc_snapshot(main_identity["pid"])
        if main_now["boot_id"] != main_identity["boot_id"] or main_now["start"] != main_identity["start"]:
            raise QualityFailure("main server identity changed while proving positive marker")
        worker = _proc_snapshot(marker_pid)
        if worker["boot_id"] != main_identity["boot_id"] or worker["state"] == "Z":
            continue
        if marker_pid != main_identity["pid"] and worker["comm"] == "resource_tracker":
            continue
        if marker_pid != main_identity["pid"] and not _ancestry_reaches_main(marker_pid, main_identity["pid"], main_identity["boot_id"]):
            continue
        main_after = _proc_snapshot(main_identity["pid"])
        if main_after["boot_id"] != main_identity["boot_id"] or main_after["start"] != main_identity["start"]:
            raise QualityFailure("main server identity changed during positive-marker proof")
        return {
            "marker": {
                "pid": marker_pid,
                "bs": value["bs"],
                "draft_tokens": value["draft_tokens"],
                "shape": value["shape"],
                "observer_removed": value["observer_removed"],
            },
            "worker": {"pid": worker["pid"], "start": worker["start"]},
            "main": {
                "pid": main_identity["pid"],
                "start": main_identity["start"],
                "boot_id": main_identity["boot_id"],
            },
        }
    raise QualityFailure("no owned positive min-p marker was appended after synthetic requests")


def _server_info(opener: Any, handle: Any) -> None:
    request = urllib.request.Request(URL + "/get_server_info", method="GET", headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            info = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TypeError) as exc:
        raise QualityFailure(f"server-info preflight failed: {exc}") from exc
    if not isinstance(info, dict) or "server_args" in info:
        raise QualityFailure("/get_server_info must be a flat object")
    context_length = info.get("context_length")
    max_total_num_tokens = info.get("max_total_num_tokens")
    if (
        info.get("status") != "ready"
        or info.get("served_model_name") != EXPECTED_MODEL
        or info.get("speculative_algorithm") != "DFLASH"
        or info.get("speculative_dflash_block_size") != 8
        or isinstance(info.get("speculative_accept_threshold_acc"), bool)
        or info.get("speculative_accept_threshold_acc") != 1.0
        or isinstance(info.get("speculative_accept_threshold_single"), bool)
        or info.get("speculative_accept_threshold_single") != 1.0
        or isinstance(context_length, bool)
        or not isinstance(context_length, (int, float))
        or context_length < 131072
        or isinstance(max_total_num_tokens, bool)
        or not isinstance(max_total_num_tokens, (int, float))
        or max_total_num_tokens < 131072
    ):
        raise QualityFailure("server READY/DFLASH contract or capacity is invalid")
    # Record only non-secret scalar identity/capacity fields; never dump api_key data.
    safe = {
        key: info[key]
        for key in (
            "status", "served_model_name", "speculative_algorithm",
            "speculative_dflash_block_size", "speculative_accept_threshold_acc",
            "speculative_accept_threshold_single", "context_length", "max_total_num_tokens",
        )
        if key in info and isinstance(info[key], (str, int, float, bool, type(None)))
    }
    _write(handle, {"record_type": "server_info", "passed": True, "metadata": safe})


def _run() -> int:
    main_identity, log_before = _preflight()
    handle = os.fdopen(os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8")
    opener = build_isolated_opener()
    try:
        try:
            _server_info(opener, handle)
            request = urllib.request.Request(URL + "/v1/models", method="GET", headers={"Accept": "application/json"})
            with opener.open(request, timeout=TIMEOUT) as response:
                model_response = json.load(response)
            model_ids = [item.get("id") for item in model_response.get("data", []) if isinstance(item, dict)]
            if EXPECTED_MODEL not in model_ids:
                raise QualityFailure(f"expected model identity not served: {EXPECTED_MODEL}")
            _write(handle, {"record_type": "preflight", "passed": True, "model_ids": model_ids})
            reasoning_seen = False
            for name, prompt, expected in (
                ("product", "Compute 987 times 123. Check your arithmetic. Return final JSON with key product.", {"product": 121401}),
                ("gcd", "Use the Euclidean algorithm to find gcd(1071,462). Return final JSON with key gcd.", {"gcd": 21}),
                ("probability", "A bag has 5 red and 3 blue balls. Draw 2 without replacement. What is probability both red? Return final JSON with key probability and a reduced fraction string.", {"probability": "5/14"}),
                ("crt", "Find the smallest nonnegative integer x with x mod 5=2 and x mod 7=3. Return final JSON with key x.", {"x": 17}),
            ):
                reasoning_seen = _arithmetic(handle, opener, name, prompt, expected) or reasoning_seen
            if not reasoning_seen:
                raise QualityFailure("no arithmetic reasoning was visible")
            _tool_roundtrip(handle, opener)
            _seed_cases(handle, opener)
            _optional_pair(handle, opener)
            proof = _require_post_test_marker(log_before, main_identity)
            _write(handle, {"record_type": "positive_marker_proof", "passed": True, **proof})
            _write(handle, {"record_type": "summary", "passed": True, "claim_scope": "final synthetic quality/integration; performance is measured separately after observer removal"})
            return 0
        except Exception as exc:
            _write(handle, {"record_type": "summary", "passed": False, "error": str(exc)})
            return 1
    finally:
        handle.close()


def main(argv: list[str] | None = None) -> int:
    _parser().parse_args(argv)
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
