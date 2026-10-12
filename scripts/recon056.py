"""Read-only final service identity, equivalence, protection, and stability proof."""
from pathlib import Path
import hashlib,json,os,subprocess,time,urllib.request
b=Path('/home/operator/qwen38-5090/optimization-runs');r=b/'pi-serving-056';candidate=b/'exp048-kda2'
def process_identity():
 o=json.loads((r/'server-pid.json').read_text());p=Path('/proc')/str(o['pid']);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (p/'stat').read_text().split()[21]==o['stat'].split()[21] and b'qwen3.8-27b-normal' in (p/'cmdline').read_bytes();return {'pid':o['pid'],'start':o['stat'].split()[21],'boot':o['boot_id']}
def opener():return urllib.request.build_opener(urllib.request.ProxyHandler({}))
def api():
 op=opener();info=json.loads(op.open('http://127.0.0.1:8081/get_server_info',timeout=10).read());models=json.loads(op.open('http://127.0.0.1:8081/v1/models',timeout=10).read());metrics=op.open('http://127.0.0.1:8081/metrics',timeout=10).read(2097152).decode();counts=[x for x in metrics.splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert len(counts)==2 and all(float(x.rsplit(' ',1)[1])==0 for x in counts);expected={'status':'ready','served_model_name':'qwen3.8-27b-normal','context_length':131200,'max_total_num_tokens':131328,'attention_backend':'flashinfer','decode_attention_backend':None,'speculative_algorithm':'DFLASH','speculative_dflash_block_size':8,'speculative_accept_threshold_single':1.0,'speculative_accept_threshold_acc':1.0,'fp4_gemm_runner_backend':'flashinfer_cutlass'};actual={k:info.get(k) for k in expected};assert actual==expected and any(x.get('id')=='qwen3.8-27b-normal' for x in models['data']);return actual
def digest_tree(root):
 out={}
 for parent,ds,fs in os.walk(root,followlinks=False):
  for name in ds+fs:assert not (Path(parent)/name).is_symlink()
  for name in fs:
   p=Path(parent)/name;out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return out
before=json.loads((r/'protected-cache-before.json').read_text());assert {k:digest_tree(Path(path)) for k,path in before['paths'].items()}==before['hashes']
for path,want in {'/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/srt/speculative/dflash_utils.py':'fb72a3a28d5ecb33d77c6cd6655cc5b4abc8086feba5fb879c44b98b5cbb8374','/home/operator/.venvs/qwen38/lib/python3.12/site-packages/sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py':'b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5'}.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==want
assert not (r/'dflash_minp035.py').exists()
def normalized_launch(root):
 m=json.loads((root/'launch.json').read_text());m.pop('wait_for',None);m['cwd']='__ROOT__';m['port']='__PORT__';argv=m['argv'];argv[argv.index('--port')+1]='__PORT__';argv[argv.index('--served-model-name')+1]='__MODEL__'
 for flag in ['--file-storage-path','--crash-dump-folder','--download-dir']:argv[argv.index(flag)+1]='__ROOT_PATH__'
 text=json.dumps(m,sort_keys=True).replace(str(root),'__ROOT__').replace('/tmp/qwen-'+root.name,'__TMPROOT__');return json.loads(text)
assert normalized_launch(r)==normalized_launch(candidate)
a=(candidate/'sitecustomize-kda048.py').read_text().replace('exp048-kda2','ROOT').replace('EXP048','LABEL').replace('KDA048','KDA_LABEL')
z=(r/'sitecustomize-kda056.py').read_text().replace('pi-serving-056','ROOT').replace('FINAL056','LABEL').replace('KDA056','KDA_LABEL');assert a==z and (r/'sitecustomize.py').read_text()==(r/'sitecustomize-kda056.py').read_text()
quality=[json.loads(x) for x in (r/'results/quality-api.jsonl').read_text().splitlines()];assert quality and all(x.get('passed') is True for x in quality)
win=Path('/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows/pi-serving-056/stderr.log').read_text();markers=[x for x in win.splitlines() if 'FINAL056_KDA2_ENABLED' in x];assert markers
first={'identity':process_identity(),'api':api()};start=time.monotonic();time.sleep(125);second={'identity':process_identity(),'api':api(),'elapsed':time.monotonic()-start};assert first['identity']==second['identity']
free,power=map(float,subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free,power.limit','--format=csv,noheader,nounits'],text=True).strip().split(','));assert power==460
print(json.dumps({'status':'PASS','first':first,'second':second,'launch_matches_validated_candidate':True,'hook_matches_validated_candidate':True,'kda_startup_markers':len(markers),'quality_records':len(quality),'protected_cache_unchanged':True,'installed_sources_unchanged':True,'minp_not_installed':True,'gpu_free_MiB':free,'power_limit_W':power}))
