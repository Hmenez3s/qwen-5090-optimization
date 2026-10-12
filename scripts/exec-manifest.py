"""Foreground WSL experiment runner; Windows controller owns its lifetime."""
from pathlib import Path
import hashlib,json,os,socket,subprocess,sys,time
def ensure_owned_tmp(raw):
    tmp=Path(raw)
    assert tmp.is_absolute() and tmp.parts[:2]==('/', 'tmp')
    assert len(tmp.parts) in (3,4) and tmp.parts[2].startswith('qwen-')
    assert '..' not in tmp.parts and len(os.fsencode(str(tmp)))<60
    for directory in [Path(*tmp.parts[:3]),tmp]:
        assert not directory.is_symlink(), 'TMPDIR symlink forbidden'
        if directory.exists():
            assert directory.is_dir() and directory.stat().st_uid==os.getuid()
        else:
            directory.mkdir()
    assert tmp.resolve()==tmp
    return tmp

m=json.loads(Path(sys.argv[1]).read_text())
root=Path(m['cwd'])
assert root.is_dir() and str(root).startswith('/home/operator/qwen38-5090/optimization-runs/')
binary=m['argv'][0]
if binary!='/home/operator/.venvs/qwen38/bin/python':
    assert binary=='/home/operator/qwen38-5090/optimization-runs/ninfer-build-014/build-c/apps/ninfer-serve'
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest()=='fbf2b0e645e1dd96fc25d173d425ab65939febac33cb664cd0c395f7bd215d8a'
assert '--delete-ckpt-after-loading' not in m['argv']
assert m['environment']['HF_HUB_OFFLINE']=='1'
assert m['environment']['PYTHONDONTWRITEBYTECODE']=='1'
assert m['environment']['TMPDIR'].startswith('/tmp/qwen-')
with (root/'server-pid.json').open('x') as f:json.dump({'pid':os.getpid(),'stat':Path('/proc/self/stat').read_text(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},f)
start=time.monotonic()
old=m.get('wait_for')
print(json.dumps({'stage':'waiting','old':old}),flush=True)
while old:
    p=Path('/proc')/str(old['pid'])
    same_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
    try: alive=same_boot and p.exists() and (p/'stat').read_text().split()[21]==old['start_ticks']
    except FileNotFoundError: alive=False
    if not alive:break
    if time.monotonic()-start>300:raise TimeoutError('original server did not stop within300s')
    time.sleep(1)
if m.get('port') is not None:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',m['port']))
free=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
assert len(free)==1 and int(free[0])>=27000, 'GPU not exclusively available'
ensure_owned_tmp(m['environment']['TMPDIR'])
os.chdir(root)
print(json.dumps({'stage':'exec','port':m['port'],'pid':os.getpid()}),flush=True)
os.execve(m['argv'][0],m['argv'],m['environment'])
