import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_stream as bench


MODEL_BODY = json.dumps({"data": [{"id": "run-owned-qwen"}]}).encode()

# Meta-only fixture grounded in /var/folders/q4/g0d2x3rn1kn6hwl97mhp86nh0000gp/T/qwen-optimization-evidence-egcfngsc/audit-lmhead4-short.json.
HISTORICAL_AUDIT_META = {
    "finish_reason": {"type": "length", "length": 512},
    "prompt_tokens": 19,
    "completion_tokens": 512,
    "cached_tokens": 0,
    "spec_accept_length": 5.953488372093023,
    "spec_verify_ct": 86,
}


def event(count, *, text="", finish="stop", include_usage=True, include_spec=True):
    meta = {"completion_tokens": count}
    if include_usage:
        meta.update({
            "prompt_tokens": 4,
            "cached_tokens": 0,
            "finish_reason": finish,
        })
        if include_spec:
            meta.update({
                "spec_accept_length": 1,
                "spec_verify_ct": 1,
            })
    return ("data: " + json.dumps({"text": text, "meta_info": meta}) + "\n").encode()


def meta_event(meta, *, text=""):
    return ("data: " + json.dumps({"text": text, "meta_info": meta}) + "\n").encode()


def openai_event(delta=None, *, finish=None, usage=None):
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        "usage": usage,
    }
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def openai_usage_event(completion_tokens=4):
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"prompt_tokens": 6, "completion_tokens": completion_tokens, "total_tokens": 10},
    }
    return ("data: " + json.dumps(payload) + "\n\n").encode()


class FakeResponse:
    def __init__(self, lines=(), status=200, body=b""):
        self.lines = list(lines)
        self.status = status
        self.body = body
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.body

    def close(self):
        self.closed = True


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected network request")
        return self.responses.pop(0)


def valid_manifest(payload=None, repetitions=1):
    return {
        "cases": [{"name": "synthetic", "payload": payload or {"text": "hello", "sampling_params": {"temperature": 0}}}],
        "repetitions": repetitions,
    }


def valid_openai_payload():
    return {
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 32,
    }


def openai_stream_response(completion_tokens=4, *, lines=None):
    if lines is None:
        lines = [
            openai_event({"role": "assistant", "content": ""}),
            openai_event({"reasoning_content": "think"}),
            openai_event({"content": "answer"}),
            openai_event({}, finish="stop"),
            openai_usage_event(completion_tokens),
            b"data: [DONE]\n\n",
        ]
    return FakeResponse(lines)


def stream_response(final_count=3, *, lines=None):
    if lines is None:
        lines = [b"data: \n", event(0), event(1, text="a"), event(final_count, text="answer"), b"data: [DONE]\n"]
    return FakeResponse(lines)


class NullableDeltaTests(unittest.TestCase):
    def test_null_role_delta_is_not_a_token_but_nonstring_still_fails(self):
        lines = [openai_event({"role": "assistant", "reasoning_content": None, "content": None}),
                 openai_event({"reasoning_content": "The"}),
                 openai_event({"content": "42"}, finish="stop"), openai_usage_event(4)]
        result = bench.measure_stream(FakeResponse(lines), 0, iter([1, 2, 3, 4, 5]).__next__, "openai")
        self.assertEqual(result["metrics"]["ttft_seconds"], 2)
        self.assertEqual(result["reasoning"], "The")
        self.assertEqual(result["text"], "42")
        with self.assertRaises(bench.StreamProtocolError):
            bench.measure_stream(FakeResponse([openai_event({"content": False})]), 0, lambda: 1, "openai")


class ValidationTests(unittest.TestCase):
    def test_url_is_literal_and_shared_ports_are_rejected(self):
        for value in ("http://localhost:3000", "https://127.0.0.1:3000", "http://127.0.0.1:8081", "http://127.0.0.1:8002", "http://127.0.0.1:3000/", "http://127.0.0.1:3000/v1"):
            with self.subTest(value=value):
                with self.assertRaises(bench.ValidationError):
                    bench.validate_url(value)
        self.assertEqual(bench.validate_url("http://127.0.0.1:3000"), "http://127.0.0.1:3000")

    def test_manifest_and_payload_validation(self):
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest({"cases": [], "repetitions": 1})
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest(valid_manifest(repetitions=0))
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest(valid_manifest({"text": "x", "stream": True, "sampling_params": {}}))
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest(valid_manifest({"text": "x", "sampling_params": {}, "image_url": "https://example.invalid/x"}))
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest(valid_manifest({"text": "x", "sampling_params": {"session": "remote"}}))
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest(valid_manifest({"text": "x", "sampling_params": {"load_path": "https://example.invalid"}}))
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest(valid_manifest({"text": "x", "input_ids": [1], "sampling_params": {}}))
        accepted = bench.validate_manifest(valid_manifest({"input_ids": [1, 2], "sampling_params": {}, "return_logprob": False}))
        self.assertEqual(accepted["cases"][0]["payload"]["input_ids"], [1, 2])

    def test_openai_payload_is_restricted_and_sampling_is_bounded(self):
        accepted = bench.validate_manifest(valid_manifest(valid_openai_payload()), "openai")
        self.assertEqual(accepted["cases"][0]["payload"], valid_openai_payload())
        for extra in ({"tools": []}, {"images": []}, {"files": []}, {"model": "identity"}, {"stream": True}):
            with self.subTest(extra=extra):
                with self.assertRaises(bench.ValidationError):
                    bench.validate_manifest(valid_manifest({**valid_openai_payload(), **extra}), "openai")
        for field, value in (("temperature", float("nan")), ("temperature", 2.1), ("top_p", 1.1), ("top_p", 0.0), ("max_tokens", 0), ("max_tokens", 131073)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(bench.ValidationError):
                    bench.validate_manifest(valid_manifest({**valid_openai_payload(), field: value}), "openai")
        with self.assertRaises(bench.ValidationError):
            bench.validate_manifest(valid_manifest({**valid_openai_payload(), "messages": [{"role": "user", "content": ["image"]}]}), "openai")

    def test_existing_output_refused_before_opener(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            output.write_text("keep\n", encoding="utf-8")
            opener = FakeOpener([])
            with self.assertRaises(bench.ValidationError):
                bench.run_benchmark(url="http://127.0.0.1:3000", expected_model="run-owned-qwen", manifest=valid_manifest(), output=output, opener=opener)
            self.assertEqual(opener.requests, [])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep\n")


class OpenerSafetyTests(unittest.TestCase):
    def test_opener_has_empty_proxy_and_no_redirect_handler(self):
        with patch.object(bench.urllib.request, "build_opener", wraps=bench.urllib.request.build_opener) as build:
            opener = bench.build_isolated_opener()
        self.assertIsInstance(build.call_args.args[0], bench.urllib.request.ProxyHandler)
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertTrue(any(isinstance(handler, bench.NoRedirectHandler) for handler in opener.handlers))
        self.assertFalse(any(type(handler) is bench.urllib.request.HTTPRedirectHandler for handler in opener.handlers))


class StreamMetricTests(unittest.TestCase):
    def test_first_positive_chunk_is_excluded_from_decode(self):
        response = stream_response()
        ticks = iter([10.0, 10.1, 10.2, 10.4, 10.5, 10.5])
        result = bench.measure_stream(response, 10.0, lambda: next(ticks))
        self.assertAlmostEqual(result["metrics"]["ttft_seconds"], 0.2)
        self.assertAlmostEqual(result["metrics"]["decode_tokens_per_second"], 10.0)
        self.assertAlmostEqual(result["metrics"]["e2e_tokens_per_second"], 6.0)
        self.assertEqual(result["events"][0]["completion_tokens"], None)
        self.assertEqual(result["meta"]["finish_reason"], "stop")
        self.assertEqual(result["text"], "answer")

    def test_empty_initial_event_is_preserved(self):
        response = stream_response(lines=[b"data: \n", event(1), event(2, text="done"), b"data: [DONE]\n"])
        ticks = iter([1.0, 1.1, 1.2, 1.3, 1.4])
        result = bench.measure_stream(response, 1.0, lambda: next(ticks))
        self.assertEqual(result["events"][0]["raw_sse"], "data: ")
        self.assertEqual(result["events"][0]["completion_tokens"], None)

    def test_grounded_native_typed_finish_reason_is_preserved(self):
        response = stream_response(lines=[event(1), meta_event(HISTORICAL_AUDIT_META, text="answer")])
        ticks = iter([2.0, 2.1, 2.3])
        result = bench.measure_stream(response, 2.0, lambda: next(ticks))
        self.assertEqual(result["meta"], HISTORICAL_AUDIT_META)
        self.assertEqual(result["finish_reason"], {"type": "length", "length": 512})
        self.assertEqual(result["finish_reason_type"], "length")

    def test_target_only_meta_without_optional_spec_fields_is_valid(self):
        response = stream_response(lines=[event(1, include_spec=False), event(2, text="done", include_spec=False)])
        ticks = iter([2.5, 2.6, 2.8])
        result = bench.measure_stream(response, 2.5, lambda: next(ticks))
        self.assertNotIn("spec_accept_length", result["meta"])
        self.assertNotIn("spec_verify_ct", result["meta"])
        self.assertIsNotNone(result["metrics"]["decode_tokens_per_second"])

    def test_delayed_duplicate_count_does_not_extend_decode_window(self):
        final_meta = {"completion_tokens": 3, "prompt_tokens": 4, "cached_tokens": 0, "finish_reason": "stop"}
        response = stream_response(lines=[event(1), event(3, text="partial"), meta_event(final_meta, text="answer")])
        ticks = iter([6.1, 6.3, 8.0, 8.5])
        result = bench.measure_stream(response, 6.0, lambda: next(ticks))
        self.assertAlmostEqual(result["metrics"]["decode_tokens_per_second"], 10.0)
        self.assertAlmostEqual(result["metrics"]["e2e_tokens_per_second"], 1.2)

    def test_nonmonotonic_counts_are_error(self):
        response = stream_response(lines=[event(1), event(0, finish="length")])
        ticks = iter([2.0, 2.1, 2.2])
        with self.assertRaises(bench.StreamProtocolError) as caught:
            bench.measure_stream(response, 2.0, lambda: next(ticks))
        self.assertIn("not monotonic", str(caught.exception))
        self.assertEqual(len(caught.exception.events), 1)

    def test_native_abort_reports_cause_before_missing_usage(self):
        meta = {"completion_tokens": 1012, "finish_reason": {"type": "abort", "message": "Out of memory even after retracting all other requests", "status_code": 500}}
        response = stream_response(lines=[event(1), meta_event(meta)])
        ticks = iter([1.0, 1.1])
        with self.assertRaises(bench.StreamProtocolError) as caught:
            bench.measure_stream(response, 1.0, lambda: next(ticks))
        self.assertIn("server aborted request: Out of memory", str(caught.exception))
        self.assertEqual(caught.exception.final_meta, meta)

    def test_missing_usage_is_error(self):
        response = stream_response(lines=[event(1), event(2, include_usage=False)])
        ticks = iter([3.0, 3.1, 3.2, 3.3])
        with self.assertRaises(bench.StreamProtocolError) as caught:
            bench.measure_stream(response, 3.0, lambda: next(ticks))
        self.assertIn("missing usage fields", str(caught.exception))

    def test_singleton_has_null_decode_rate(self):
        response = stream_response(lines=[event(1), b"data: [DONE]\n"])
        ticks = iter([4.0, 4.4, 4.4])
        result = bench.measure_stream(response, 4.0, lambda: next(ticks))
        self.assertIsNone(result["metrics"]["decode_tokens_per_second"])

    def test_error_sse_is_error(self):
        response = stream_response(lines=[event(1), b'data: {"error":"bad"}\n'])
        ticks = iter([5.0, 5.1, 5.2])
        with self.assertRaises(bench.StreamProtocolError):
            bench.measure_stream(response, 5.0, lambda: next(ticks))

    def test_malformed_sse_json_is_error(self):
        response = stream_response(lines=[b"data: {not-json}\n"])
        with self.assertRaises(bench.StreamProtocolError) as caught:
            bench.measure_stream(response, 6.0, lambda: 6.1)
        self.assertIn("malformed SSE JSON", str(caught.exception))


class OpenAIStreamMetricTests(unittest.TestCase):
    def test_role_only_reasoning_content_and_usage_only_final(self):
        response = openai_stream_response()
        ticks = iter([10.1, 10.2, 10.3, 10.5, 10.7, 10.8, 10.9])
        result = bench.measure_stream(response, 10.0, lambda: next(ticks), "openai")
        self.assertAlmostEqual(result["metrics"]["ttft_seconds"], 0.2)
        self.assertEqual(result["text"], "answer")
        self.assertEqual(result["reasoning"], "think")
        self.assertIsNone(result["metrics"]["decode_tokens_per_second"])
        self.assertAlmostEqual(result["metrics"]["stream_tokens_per_second_estimate"], 4 / 0.1)
        self.assertAlmostEqual(result["metrics"]["e2e_tokens_per_second"], 4 / 0.9)
        self.assertEqual(result["usage"]["completion_tokens"], 4)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertAlmostEqual(result["events"][0]["seconds"], 0.1)
        self.assertTrue(any("hidden-token bias" in note for note in result["metric_notes"]))

    def test_multi_token_first_chunk_is_estimate_not_exact_decode(self):
        response = openai_stream_response(lines=[
            openai_event({"role": "assistant", "content": ""}),
            openai_event({"content": "many tokens in one delta"}),
            openai_event({}, finish="stop"),
            openai_usage_event(8),
            b"data: [DONE]\n\n",
        ])
        ticks = iter([1.1, 1.2, 1.5, 1.6, 1.7, 1.8])
        result = bench.measure_stream(response, 1.0, lambda: next(ticks), protocol="openai")
        self.assertIsNone(result["metrics"]["decode_tokens_per_second"])
        self.assertIsNone(result["metrics"]["stream_tokens_per_second_estimate"])

    def test_missing_usage_and_sse_error_are_protocol_errors(self):
        missing_usage = openai_stream_response(lines=[
            openai_event({"content": "answer"}),
            openai_event({}, finish="stop"),
            b"data: [DONE]\n\n",
        ])
        with self.assertRaises(bench.StreamProtocolError) as missing:
            bench.measure_stream(missing_usage, 2.0, lambda: 2.1, protocol="openai")
        self.assertIn("usage", str(missing.exception))
        error = openai_stream_response(lines=[b'data: {"error":{"message":"bad"}}\n\n'])
        with self.assertRaises(bench.StreamProtocolError):
            bench.measure_stream(error, 3.0, lambda: 3.1, protocol="openai")


class RunTests(unittest.TestCase):
    def test_payload_is_preserved_except_stream_and_records_hash_settings(self):
        payload = {"text": "synthetic", "sampling_params": {"temperature": 0.2, "max_new_tokens": 3}, "return_logprob": True}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            opener = FakeOpener([FakeResponse(body=MODEL_BODY), stream_response()])
            bench.run_benchmark(url="http://127.0.0.1:3000", expected_model="run-owned-qwen", manifest=valid_manifest(payload), output=output, opener=opener, clock=time_clock([1, 1.1, 1.2, 1.4, 1.5, 1.5, 1.5]))
            sent = json.loads(opener.requests[1][0].data.decode())
            self.assertEqual(sent, {**payload, "stream": True})
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
            self.assertEqual(record["declared_settings"]["sampling_params"], payload["sampling_params"])
            self.assertEqual(len(record["request_sha256"]), 64)

    def test_openai_request_identity_and_payload_are_preserved(self):
        payload = valid_openai_payload()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            opener = FakeOpener([FakeResponse(body=MODEL_BODY), openai_stream_response()])
            bench.run_benchmark(
                url="http://127.0.0.1:3000",
                expected_model="run-owned-qwen",
                manifest=valid_manifest(payload),
                output=output,
                opener=opener,
                clock=time_clock([1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.8, 1.9]),
                protocol="openai",
            )
            request = opener.requests[1][0]
            self.assertEqual(request.full_url, "http://127.0.0.1:3000/v1/chat/completions")
            sent = json.loads(request.data.decode())
            self.assertEqual(sent, {**payload, "model": "run-owned-qwen", "stream": True, "stream_options": {"include_usage": True}})
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
            self.assertEqual(record["protocol"], "openai")
            self.assertEqual(record["declared_settings"], payload)

    def test_openai_http_error_aborts_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            opener = FakeOpener([FakeResponse(body=MODEL_BODY), FakeResponse(status=503, body=b"busy")])
            with self.assertRaises(bench.BenchmarkError):
                bench.run_benchmark(
                    url="http://127.0.0.1:3000",
                    expected_model="run-owned-qwen",
                    manifest=valid_manifest(valid_openai_payload(), repetitions=3),
                    output=output,
                    opener=opener,
                    protocol="openai",
                )
            self.assertEqual(len(opener.requests), 2)
            self.assertEqual(opener.requests[1][0].full_url, "http://127.0.0.1:3000/v1/chat/completions")
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertIn("503", records[-1]["error"])
            self.assertEqual(len(records), 2)

    def test_expected_identity_mismatch_happens_before_post(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            opener = FakeOpener([FakeResponse(body=json.dumps({"data": [{"id": "other"}]}).encode())])
            with self.assertRaises(bench.BenchmarkError):
                bench.run_benchmark(url="http://127.0.0.1:3000", expected_model="run-owned-qwen", manifest=valid_manifest(), output=output, opener=opener)
            self.assertEqual(len(opener.requests), 1)
            self.assertEqual(opener.requests[0][0].method, "GET")
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["phase"], "model_preflight")

    def test_http_error_and_error_sse_abort_remaining_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            error_response = stream_response(lines=[b'data: {"error":{"message":"stop"}}\n'])
            opener = FakeOpener([FakeResponse(body=MODEL_BODY), error_response])
            with self.assertRaises(bench.BenchmarkError):
                bench.run_benchmark(url="http://127.0.0.1:3000", expected_model="run-owned-qwen", manifest=valid_manifest(repetitions=3), output=output, opener=opener, clock=time_clock([1, 1.1, 1.2]))
            self.assertEqual(len(opener.requests), 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[-1]["status"], "error")
            self.assertEqual(len(records), 2)

    def test_http_status_error_aborts_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            opener = FakeOpener([FakeResponse(status=503, body=b"busy")])
            with self.assertRaises(bench.BenchmarkError):
                bench.run_benchmark(url="http://127.0.0.1:3000", expected_model="run-owned-qwen", manifest=valid_manifest(), output=output, opener=opener)
            self.assertEqual(len(opener.requests), 1)
            self.assertIn("503", output.read_text(encoding="utf-8"))


def time_clock(values):
    iterator = iter(values)
    return lambda: next(iterator)


if __name__ == "__main__":
    unittest.main()
