"""Private exp048: verified AOT plus two bitwise-validated KDA shapes."""
from pathlib import Path
import hashlib,os,runpy,sys
try:
    root=Path(__file__).resolve().parent
    expected_root=Path('/home/operator/qwen38-5090/optimization-runs/exp048-kda2')
    if root!=expected_root or root.is_symlink() or root.resolve()!=root:
        raise RuntimeError('KDA048 hook outside owned root')
    runpy.run_path(str(root/'sitecustomize-prebuilt032.py'),run_name='__sitecustomize_prebuilt032__')
    import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm as gate
    expected='b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5'
    actual=hashlib.sha256(Path(gate.__file__).read_bytes()).hexdigest()
    if actual!=expected:
        raise RuntimeError('KDA gate source hash mismatch '+actual)
    added={(8,17408,5120),(8,5120,248320)}
    if gate._SUPPORTED_SHAPES.intersection(added):
        raise RuntimeError('KDA048 baseline eligibility changed')
    gate._SUPPORTED_SHAPES=gate._SUPPORTED_SHAPES.union(added)
    print('EXP048_KDA2_ENABLED pid='+str(os.getpid())+' shapes='+repr(sorted(added)),file=sys.stderr,flush=True)
except Exception as exc:
    raise SystemExit('KDA048 hook failed: '+str(exc)) from exc
