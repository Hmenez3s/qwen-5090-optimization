"""Canonical private SGLang readiness watcher; no inference or launch retries."""
import json
import sys
import time
import urllib.request
from pathlib import Path


def alive(receipt):
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != receipt['boot_id']:
        return False
    try:
        stat = (Path('/proc') / str(receipt['pid']) / 'stat').read_text().split()
    except FileNotFoundError:
        return False
    expected = receipt.get('start_ticks') or receipt['stat'].split()[21]
    return stat[21] == expected and stat[2] != 'Z'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('private readiness redirects forbidden')


def main(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    root = Path(m['cwd']).resolve()
    assert root.is_relative_to('/home/operator/qwen38-5090/optimization-runs')
    assert m['port'] in (34731, 8081)
    expected = m['argv'][m['argv'].index('--served-model-name') + 1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    started = time.monotonic()
    while time.monotonic() - started < 900:
        receipt_path = root / 'server-pid.json'
        if not receipt_path.exists():
            if time.monotonic() - started >= 30:
                raise RuntimeError('launcher receipt absent after30s')
            time.sleep(.5)
            continue
        receipt = json.loads(receipt_path.read_text())
        if not alive(receipt):
            raise RuntimeError('launcher/server exited or identity changed')
        old = m.get('wait_for')
        if old and alive(old):
            time.sleep(.5)
            continue
        try:
            with opener.open(f"http://127.0.0.1:{m['port']}/server_info", timeout=3) as response:
                info = json.load(response)
        except (OSError, ValueError):
            time.sleep(2)
            continue
        if info.get('served_model_name') != expected:
            raise RuntimeError('unexpected service identity on private port')
        if info.get('status') == 'ready':
            print(json.dumps({'status':'READY','model':expected,'elapsed':time.monotonic()-started}),flush=True)
            return 0
        time.sleep(2)
    raise TimeoutError('private service not ready after900s')


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as exc:
        print('NOT_READY: '+repr(exc),file=sys.stderr,flush=True)
        sys.exit(1)
