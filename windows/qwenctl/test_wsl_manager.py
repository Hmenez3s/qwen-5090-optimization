import hashlib
import io
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import wsl_manager as wm


def validated_argv():
    return [
        "/home/operator/.venvs/qwen38/bin/python", "-m", "sglang.launch_server",
        "--model-path", wm.MODEL_PATH,
        "--served-model-name", wm.MODEL_NAME,
        "--host", "127.0.0.1", "--port", "8081",
        "--quantization", "modelopt_fp4", "--fp4-gemm-backend", "flashinfer_cutlass",
        "--speculative-algorithm", "DFLASH", "--speculative-num-steps", "1",
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "8",
        "--speculative-draft-model-quantization", "modelopt_fp4",
        "--speculative-draft-model-path", wm.DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size", "8", "--enable-single-batch-overlap",
        "--attention-backend", "flashinfer", "--kv-cache-dtype", "fp8_e4m3",
        "--max-running-requests", "1", "--max-total-tokens", "131328",
        "--chunked-prefill-size", "4096", "--cuda-graph-max-bs-decode", "1",
        "--cuda-graph-bs-decode", "1", "--disable-prefill-cuda-graph",
        "--mem-fraction-static", "0.92", "--mamba-ssm-dtype", "bfloat16",
        "--max-mamba-cache-size", "5", "--context-length", "131200",
        "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
        "--enable-metrics", "--load-format", "safetensors", "--random-seed", "7319",
    ]


class FreshRootTests(unittest.TestCase):
    def make_template(self, parent: Path) -> Path:
        template = parent / "pi-serving-059"
        template.mkdir()
        aot_dir = template / "home/.cache/flashinfer/0.6.18/120f/cached_ops"
        aot_dir.mkdir(parents=True)
        hashes = {}
        for index in range(9):
            relative = Path(f"kernel{index}/library{index}.so")
            path = aot_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"tiny-aot-{index}".encode()
            path.write_bytes(payload)
            hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
        launch = {
            "cwd": str(template),
            "port": 8081,
            "argv": validated_argv() + [
                "--file-storage-path", str(template / "storage"),
                "--crash-dump-folder", str(template / "crash"),
                "--download-dir", str(template / "downloads"),
            ],
            "environment": {
                "HOME": str(template / "home"),
                "HF_HOME": str(template / "home/huggingface"),
                "XDG_CACHE_HOME": str(template / "home/.cache"),
                "CUDA_CACHE_PATH": str(template / "cuda-cache"),
                "CUTE_DSL_CACHE_DIR": str(template / "cute-cache"),
                "CUTE_DSL_DUMP_DIR": str(template / "cute-dumps"),
                "SGLANG_CACHE_DIR": str(template / "home/.cache/sglang"),
                "TMPDIR": "/tmp/qwen-template/tmp",
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "wait_for": {"pid": 4, "boot_id": "old", "start_ticks": "1"},
        }
        (template / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
        (template / "prebuilt-aot.json").write_text(
            json.dumps(
                {
                    "directory": str(aot_dir),
                    "sha256": hashes,
                    "flashinfer_version": wm.FLASHINFER_VERSION,
                    "architecture": wm.FLASHINFER_ARCHITECTURE,
                }
            ),
            encoding="utf-8",
        )
        kda = (
            "expected_root=Path(%r)\n"
            "added={(8,17408,5120),(8,5120,248320)}\n"
            "print('FINAL059_KDA2_ENABLED')\n" % str(template)
        )
        for name in (
            "exec-manifest.py",
            "wait-ready024.py",
            "sitecustomize-prebuilt032.py",
            "sitecustomize-kda059.py",
        ):
            (template / name).write_text(kda if name.endswith("kda059.py") else "# helper\n", encoding="utf-8")
        (template / "sitecustomize.py").write_text(kda, encoding="utf-8")
        return template

    def test_prepare_copies_nine_files_and_rewrites_only_owned_values(self):
        with tempfile.TemporaryDirectory(prefix="qwenctl-test-") as raw:
            parent = Path(raw)
            template = self.make_template(parent)
            result = wm.prepare_fresh_root(
                template,
                {"context-length": 131180, "max-total-tokens": 131300, "random-seed": 17, "startup-timeout-seconds": 900},
                base_root=parent,
                name_factory=lambda: "qwenctl-20260910T010203Z-1234abcd",
            )
            root = Path(result["root"])
            self.assertEqual(root.name, "qwenctl-20260910T010203Z-1234abcd")
            self.assertFalse((root / "server-pid.json").exists())
            self.assertFalse((root / "launch.json").read_text(encoding="utf-8").find("wait_for") >= 0)
            launch = json.loads((root / "launch.json").read_text(encoding="utf-8"))
            self.assertEqual(launch["cwd"], str(root))
            self.assertEqual(launch["argv"][launch["argv"].index("--context-length") + 1], "131180")
            self.assertEqual(launch["argv"][launch["argv"].index("--random-seed") + 1], "17")
            self.assertEqual(launch["environment"]["TMPDIR"], f"/tmp/qwen-{root.name}/tmp")
            self.assertEqual((root / "sitecustomize.py").read_bytes(), (root / "sitecustomize-kda059.py").read_bytes())
            self.assertIn(str(root), (root / "sitecustomize.py").read_text(encoding="utf-8"))
            self.assertNotIn(str(template), (root / "sitecustomize.py").read_text(encoding="utf-8"))
            aot = json.loads((root / "prebuilt-aot.json").read_text(encoding="utf-8"))
            self.assertEqual(len(aot["sha256"]), 9)
            for relative in aot["sha256"]:
                source = Path(template / "home/.cache/flashinfer/0.6.18/120f/cached_ops" / relative)
                copied = Path(aot["directory"]) / relative
                self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), aot["sha256"][relative])
                self.assertNotEqual((source.stat().st_dev, source.stat().st_ino), (copied.stat().st_dev, copied.stat().st_ino))
                self.assertEqual(copied.stat().st_nlink, 1)
            self.assertTrue(template.joinpath("launch.json").read_text(encoding="utf-8").find(str(root)) < 0)

    def test_missing_validated_launch_flag_is_rejected_before_root_creation(self):
        with tempfile.TemporaryDirectory(prefix="qwenctl-test-") as raw:
            parent = Path(raw)
            template = self.make_template(parent)
            launch = json.loads((template / "launch.json").read_text(encoding="utf-8"))
            index = launch["argv"].index("--fp4-gemm-backend")
            del launch["argv"][index:index + 2]
            (template / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
            with self.assertRaises(wm.ManagerError):
                wm.prepare_fresh_root(
                    template,
                    wm.CONFIG_DEFAULTS,
                    base_root=parent,
                    name_factory=lambda: "qwenctl-20260910T010203Z-1234abcd",
                )
            self.assertFalse((parent / "qwenctl-20260910T010203Z-1234abcd").exists())

    def test_aot_hash_mismatch_leaves_partial_root_and_does_not_delete_template(self):
        with tempfile.TemporaryDirectory(prefix="qwenctl-test-") as raw:
            parent = Path(raw)
            template = self.make_template(parent)
            aot = json.loads((template / "prebuilt-aot.json").read_text(encoding="utf-8"))
            first = next(iter(aot["sha256"]))
            aot["sha256"][first] = "0" * 64
            (template / "prebuilt-aot.json").write_text(json.dumps(aot), encoding="utf-8")
            with self.assertRaises(wm.ManagerError):
                wm.prepare_fresh_root(
                    template,
                    wm.CONFIG_DEFAULTS,
                    base_root=parent,
                    name_factory=lambda: "qwenctl-20260910T010203Z-1234abcd",
                )
            self.assertTrue((parent / "qwenctl-20260910T010203Z-1234abcd").exists())
            self.assertTrue((template / "launch.json").exists())


class IdentityAndStopTests(unittest.TestCase):
    def setUp(self):
        self.old_base = wm.BASE_ROOT
        self.old_template = wm.TEMPLATE_ROOT

    def tearDown(self):
        wm.BASE_ROOT = self.old_base
        wm.TEMPLATE_ROOT = self.old_template

    def make_proc(self, root: Path, pid: int = 321):
        proc = root / "proc"
        (proc / "sys/kernel/random").mkdir(parents=True)
        (proc / "sys/kernel/random/boot_id").write_text("boot-1\n", encoding="utf-8")
        (proc / str(pid)).mkdir()
        fields = [str(pid), "(server)", "S"] + ["0"] * 19
        fields[21] = "777"
        (proc / str(pid) / "stat").write_text(" ".join(fields), encoding="utf-8")
        (proc / str(pid) / "cmdline").write_bytes((wm.MODEL_NAME + "\0--port\08081\0").encode())
        return proc

    def make_running_root(self, parent: Path, proc: Path):
        base = parent / "optimization-runs"
        template = base / "pi-serving-059"
        template.mkdir(parents=True)
        argv = validated_argv()
        launch = {"cwd": str(template), "port": 8081, "argv": argv, "environment": {"TMPDIR": "/tmp/qwen-x", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}}
        (template / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
        (template / "server-pid.json").write_text(
            json.dumps({"pid": 321, "boot_id": "boot-1", "start_ticks": "777"}), encoding="utf-8"
        )
        return base, template

    def test_metrics_require_one_numeric_zero_sample_per_metric(self):
        text = "sglang:num_running_reqs{} 0\nsglang:num_queue_reqs{} 0\n"
        self.assertTrue(wm.metrics_are_idle(text))
        self.assertFalse(wm.metrics_are_idle(text + "sglang:num_queue_reqs{} 0\n"))
        self.assertFalse(wm.metrics_are_idle("sglang:num_running_reqs{} NaN\nsglang:num_queue_reqs{} 0\n"))

    def test_stop_sends_one_sigterm_only_after_all_identity_guards(self):
        with tempfile.TemporaryDirectory(prefix="qwenctl-test-") as raw:
            parent = Path(raw)
            proc = self.make_proc(parent)
            base, root = self.make_running_root(parent, proc)
            wm.BASE_ROOT = base
            wm.TEMPLATE_ROOT = root
            token = {"root": str(root), "pid": 321, "boot_id": "boot-1", "start_ticks": "777", "model": wm.MODEL_NAME, "port": 8081, "controller_pid": 44}
            controller = {"count": 1, "matches": [{"pid": 44, "name": "cmd.exe", "command_line": f"cmd.exe /d /c wsl.exe {root}/exec-manifest.py {root}/launch.json"}]}
            signals = []
            clock_values = iter([0.0, 1.0])

            def kill(pid, number):
                signals.append((pid, number))
                (proc / str(pid) / "stat").unlink()

            result = wm.stop_service(
                token,
                controller,
                proc_root=proc,
                api_reader=lambda port: {"served_model_name": wm.MODEL_NAME},
                metrics_reader=lambda port: "sglang:num_running_reqs{} 0\nsglang:num_queue_reqs{} 0\n",
                kill_fn=kill,
                sleep_fn=lambda _: None,
                monotonic_fn=lambda: next(clock_values),
            )
            self.assertEqual(result["signal"], "SIGTERM")
            self.assertEqual(signals, [(321, signal.SIGTERM)])

    def test_stop_rejects_missing_controller_pid_before_signal(self):
        token = {
            "root": str(wm.TEMPLATE_ROOT), "pid": 321, "boot_id": "boot-1",
            "start_ticks": "777", "model": wm.MODEL_NAME, "port": 8081,
        }
        controller = {
            "count": 1,
            "matches": [{"pid": 42, "name": "cmd.exe", "command_line": f"cmd.exe {wm.TEMPLATE_ROOT}/exec-manifest.py {wm.TEMPLATE_ROOT}/launch.json"}],
        }
        with self.assertRaises(wm.ManagerError) as raised:
            wm.stop_service(token, controller, kill_fn=lambda *_: self.fail("signal should not happen"))
        self.assertEqual(raised.exception.code, "BLOCKED")
        self.assertEqual(raised.exception.kind, "controller")

    def test_busy_metrics_never_signal(self):
        with tempfile.TemporaryDirectory(prefix="qwenctl-test-") as raw:
            parent = Path(raw)
            proc = self.make_proc(parent)
            base, root = self.make_running_root(parent, proc)
            wm.BASE_ROOT = base
            wm.TEMPLATE_ROOT = root
            token = {"root": str(root), "pid": 321, "boot_id": "boot-1", "start_ticks": "777", "model": wm.MODEL_NAME, "port": 8081, "controller_pid": 44}
            controller = {"count": 1, "matches": [{"pid": 44, "name": "cmd.exe", "command_line": f"cmd.exe {root}/exec-manifest.py {root}/launch.json"}]}
            signals = []
            with self.assertRaises(wm.ManagerError) as raised:
                wm.stop_service(
                    token,
                    controller,
                    proc_root=proc,
                    api_reader=lambda port: {"served_model_name": wm.MODEL_NAME},
                    metrics_reader=lambda port: "sglang:num_running_reqs{} 1\nsglang:num_queue_reqs{} 0\n",
                    kill_fn=lambda pid, number: signals.append((pid, number)),
                )
            self.assertEqual(raised.exception.code, "BUSY")
            self.assertEqual(signals, [])


class ProtocolTests(unittest.TestCase):
    def test_kda_source_uses_lexical_venv_when_python_is_symlink(self):
        old_template = wm.TEMPLATE_ROOT
        try:
            with tempfile.TemporaryDirectory(prefix="qwenctl-test-") as raw:
                root = Path(raw)
                interpreter = root / "venv/bin/python"
                interpreter.parent.mkdir(parents=True)
                interpreter.symlink_to(Path(sys.executable))
                source = root / "venv/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py"
                source.parent.mkdir(parents=True)
                source.write_text("# fixture\n", encoding="utf-8")
                (root / "launch.json").write_text(json.dumps({"argv": [str(interpreter)]}), encoding="utf-8")
                wm.TEMPLATE_ROOT = root
                self.assertEqual(wm._find_kda_source(), source)
        finally:
            wm.TEMPLATE_ROOT = old_template

    def test_malformed_request_is_stable_and_does_not_write(self):
        output = io.StringIO()
        wm.run_protocol(io.StringIO('{"schema_version":1,"operation":"not-allowed"}'), output)
        response = json.loads(output.getvalue())
        self.assertEqual(set(response), {"schema_version", "ok", "code", "profile", "data", "error"})
        self.assertFalse(response["ok"])
        self.assertEqual(response["code"], "INVALID_REQUEST")

    def test_config_cross_field_validation(self):
        with self.assertRaises(wm.ManagerError):
            wm.validate_config({"context-length": 131200, "max-total-tokens": 131199})
        self.assertEqual(wm.profile_for_config(wm.CONFIG_DEFAULTS), "VALIDATED")
        custom = dict(wm.CONFIG_DEFAULTS)
        custom["random-seed"] = 17
        self.assertEqual(wm.profile_for_config(custom), "CUSTOM_UNVALIDATED")


if __name__ == "__main__":
    unittest.main()
