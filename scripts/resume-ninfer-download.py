"""Task-owned ranged download with cooperative pause; no deletion/overwrite of weights."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,time,urllib.request
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-resume-013')
FILE=Path('/home/operator/qwen38-5090/models/NInfer-Normal-11dbbbbb/qwen3_8_27b_nvfp4.ninfer')
URL='https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/11dbbbbbc33db198afe2f02c9232c771ff7031be/qwen3_8_27b_nvfp4.ninfer'
SIZE=23719496192
SHA='552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c'
CHUNK=64*1024*1024
started=time.monotonic()
def status(state,**kw):
    data={'state':state,'bytes':FILE.stat().st_size,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
    (ROOT/'status.json').write_text(json.dumps(data))
    print(json.dumps(data),flush=True)
assert FILE.resolve()==FILE and FILE.stat().st_nlink==1
assert FILE.stat().st_size==3753902080, 'partial changed; investigate before appending'
(ROOT/'pid.json').write_text(json.dumps({'pid':os.getpid(),'stat':Path('/proc/self/stat').read_text(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with FILE.open('ab') as out:
        fcntl.flock(out,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while FILE.stat().st_size<SIZE:
            paused=False
            while json.loads((ROOT/'control.json').read_text()).get('pause',True):
                if not paused:status('PAUSED');paused=True
                if time.monotonic()-started>10800:raise TimeoutError('three-hour budget')
                time.sleep(1)
            offset=FILE.stat().st_size;end=min(offset+CHUNK,SIZE)-1
            if time.monotonic()-started>10800:raise TimeoutError('three-hour budget')
            status('DOWNLOADING',range_start=offset,range_end=end)
            req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-{end}','User-Agent':'QwenOptimization/1.0'})
            with opener.open(req,timeout=120) as response:
                expected=f'bytes {offset}-{end}/{SIZE}'
                assert response.status==206 and response.headers.get('Content-Range')==expected, 'range response mismatch'
                left=end-offset+1
                while left:
                    data=response.read(min(4*1024*1024,left))
                    if not data:raise EOFError('incomplete range; prefix preserved')
                    out.write(data);left-=len(data)
            out.flush();os.fdatasync(out.fileno())
            if hasattr(os,'posix_fadvise'):os.posix_fadvise(out.fileno(),offset,end-offset+1,os.POSIX_FADV_DONTNEED)
    status('HASHING')
    digest=hashlib.sha256()
    with FILE.open('rb') as inp:
        while data:=inp.read(16*1024*1024):digest.update(data)
    assert FILE.stat().st_size==SIZE and digest.hexdigest()==SHA, 'final hash mismatch'
    status('COMPLETE',sha256=digest.hexdigest())
except Exception as exc:
    status('FAILED',error=repr(exc))
    raise
