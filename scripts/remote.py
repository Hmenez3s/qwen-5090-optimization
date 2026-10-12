#!/usr/bin/env python3
"""Run a supplied Python probe on gpu-host WSL; host operations stay orchestrator-owned."""
import argparse
from pathlib import Path
import subprocess
import sys


def run(source: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run([
        'ssh', '-o', 'BatchMode=yes', '-o', 'UseKeychain=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=10', 'operator@192.0.2.12',
        'wsl.exe -d Ubuntu -- python3 -',
    ], input=source.encode(), capture_output=True, timeout=timeout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('probe', type=Path)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    result = run(args.probe.read_text(), args.timeout)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
