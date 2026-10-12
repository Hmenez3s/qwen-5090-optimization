"""Run-local AOT catalog of hash-verified, previously built FlashInfer libraries.

FlashInfer jit/env.py explicitly permits AOT scripts to override module globals.
No JitSpec method or installed source is changed; cache misses retain normal JIT.
"""
from pathlib import Path
import hashlib,json,os,sys
try:
    root=Path(__file__).resolve().parent
    if root.parent != Path('/home/operator/qwen38-5090/optimization-runs'):
        raise RuntimeError('AOT hook must run from its owned experiment root')
    manifest=json.loads((root/'prebuilt-aot.json').read_text())
    directory=Path(manifest['directory'])
    if not directory.is_relative_to(root) or directory.resolve()!=directory:
        raise RuntimeError('AOT directory outside owned root')
    for rel,expected in manifest['sha256'].items():
        p=directory/rel
        if not p.is_relative_to(directory) or p.resolve()!=p or p.suffix!='.so':
            raise RuntimeError('invalid AOT artifact path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('AOT artifact hash mismatch: '+rel)
    import flashinfer
    from flashinfer.jit import env as jit_env
    if flashinfer.__version__!=manifest['flashinfer_version']:
        raise RuntimeError('AOT FlashInfer version mismatch')
    # Enforce the same architecture catalog (120f) as the compiled libraries.
    if jit_env.FLASHINFER_WORKSPACE_DIR.name!=manifest['architecture']:
        raise RuntimeError('AOT architecture mismatch')
    jit_env.FLASHINFER_AOT_DIR=directory
    print('OWNED_PREBUILT_AOT '+str(len(manifest['sha256']))+' verified libraries',flush=True)
except Exception as exc:
    # sitecustomize ignores ordinary exceptions; SystemExit keeps this fail-closed.
    raise SystemExit('Owned AOT preflight failed: '+str(exc)) from exc
