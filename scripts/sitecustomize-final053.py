"""Final process-local AOT, KDA2, and target-only min-p correction."""
from __future__ import annotations
from pathlib import Path
import functools,hashlib,importlib.util,json,os,runpy,sys
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/pi-serving-053')
KDA_SHA='b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5'
DFLASH_SHA='fb72a3a28d5ecb33d77c6cd6655cc5b4abc8086feba5fb879c44b98b5cbb8374'
HELPER_SHA='eb43e977b6f2174819c4bc53c7f475b90f0bc5448fea6ab7e73754b4e7b4f2cb'
SHAPES={(8,17408,5120),(8,5120,248320)}
OBS='_dflash_minp053_observer'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def marker(prefix,record):print(prefix+' '+json.dumps(record,sort_keys=True,separators=(',',':')),flush=True)
def physical(name):
 p=ROOT/name
 if not p.is_file() or p.is_symlink() or p.resolve()!=p:raise RuntimeError('invalid owned file '+str(p))
 return p
def load_helper(path):
 spec=importlib.util.spec_from_file_location('_dflash_minp035_final053',path)
 if spec is None or spec.loader is None:raise RuntimeError('cannot load min-p helper')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def install():
 root=Path(__file__).resolve().parent
 if root!=ROOT or root.is_symlink() or root.resolve()!=root:raise RuntimeError('hook outside final root')
 helper_path=physical('dflash_minp035.py');aot=physical('sitecustomize-prebuilt032.py')
 if digest(helper_path)!=HELPER_SHA:raise RuntimeError('min-p helper hash mismatch')
 runpy.run_path(str(aot),run_name='__sitecustomize_prebuilt032__')
 import sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm as gate
 if digest(gate.__file__)!=KDA_SHA:raise RuntimeError('KDA gate hash mismatch')
 if gate._SUPPORTED_SHAPES.intersection(SHAPES):raise RuntimeError('final KDA baseline eligibility changed')
 gate._SUPPORTED_SHAPES=gate._SUPPORTED_SHAPES.union(SHAPES)
 from sglang.srt.speculative import dflash_utils
 if digest(dflash_utils.__file__)!=DFLASH_SHA:raise RuntimeError('dflash source hash mismatch')
 helper=load_helper(helper_path);corrected=helper.install()
 @functools.wraps(corrected)
 def observed(*args,**kwargs):
  result=corrected(*args,**kwargs)
  info=kwargs.get('sampling_info')
  if bool(getattr(info,'need_min_p_sampling',False)):
   if dflash_utils.build_dflash_verify_target_probs is observed:
    dflash_utils.build_dflash_verify_target_probs=corrected
   marker('DFLASH_MINP053_POSITIVE',{'pid':os.getpid(),'bs':int(kwargs['bs']),'draft_tokens':int(kwargs['draft_token_num']),'shape':[int(x) for x in result.shape],'observer_removed':dflash_utils.build_dflash_verify_target_probs is corrected})
  return result
 setattr(observed,OBS,True);dflash_utils.build_dflash_verify_target_probs=observed
 marker('FINAL053_STARTUP',{'pid':os.getpid(),'root':str(root),'kda_sha256':KDA_SHA,'dflash_utils_sha256':DFLASH_SHA,'helper_sha256':HELPER_SHA,'kda_shapes':[list(x) for x in sorted(SHAPES)],'minp_observer':OBS})
try:install()
except SystemExit:raise
except BaseException as exc:raise SystemExit('final053 sitecustomize failed: '+str(exc)) from exc
