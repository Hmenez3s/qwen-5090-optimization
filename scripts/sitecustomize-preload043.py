"""Private exp043: verified AOT plus existing FlashInfer tactic-cache preload."""
from pathlib import Path
import hashlib,json,runpy
try:
    root=Path(__file__).resolve().parent
    expected=Path('/home/operator/qwen38-5090/optimization-runs/exp043-split-attn-noautotune')
    if root!=expected or root.is_symlink() or root.resolve()!=root:
        raise RuntimeError('hook outside owned exp043 root')
    runpy.run_path(str(root/'sitecustomize-prebuilt032.py'),run_name='__sitecustomize_prebuilt032__')
    manifest=json.loads((root/'autotune-preload.json').read_text())
    from flashinfer.autotuner.autotuner import AutoTuner
    loaded=[]
    for rel,want in sorted(manifest['sha256'].items()):
        path=root/rel
        if not path.is_relative_to(root) or path.resolve()!=path or path.suffix!='.json':
            raise RuntimeError('invalid autotune cache path '+rel)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=want:
            raise RuntimeError('autotune cache hash mismatch '+rel)
        if not AutoTuner.get().load_configs(str(path)):
            raise RuntimeError('autotune cache metadata rejected '+rel)
        loaded.append(rel)
    if not loaded:
        raise RuntimeError('empty autotune cache preload')
    print('EXP043_PRELOADED_AUTOTUNE '+json.dumps(loaded,separators=(',',':')),flush=True)
except Exception as exc:
    raise SystemExit('exp043 preload failed: '+str(exc)) from exc
