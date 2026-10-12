from pathlib import Path,PurePosixPath
import hashlib,json,os,shutil,subprocess,tarfile,time,urllib.request
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/ninfer-build-014')
TMP=Path('/tmp/qwen-build014')
assert not ROOT.exists() and not TMP.exists()
assert shutil.disk_usage('/home/operator').free>10*1024**3
ROOT.mkdir();TMP.mkdir()
for d in ['home','uv-cache','source','apt','apt/lists','apt/lists/partial','apt/archives','apt/archives/partial','apt/conf.d','apt/log','sysroot']:(ROOT/d).mkdir(exist_ok=True)
env={'PATH':'/usr/bin:/bin','HOME':str(ROOT/'home'),'TMPDIR':str(TMP),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','UV_CACHE_DIR':str(ROOT/'uv-cache'),'PIP_CONFIG_FILE':'/dev/null','GIT_TERMINAL_PROMPT':'0','GIT_CONFIG_NOSYSTEM':'1'}
started=time.monotonic()
def run(args,timeout=900):
    print('COMMAND',json.dumps(args),flush=True)
    subprocess.run(args,cwd=ROOT,env=env,check=True,timeout=min(timeout,max(1,1800-(time.monotonic()-started))))
def state(stage,**kw):
    (ROOT/'status.json').write_text(json.dumps({'stage':stage,**kw}));print('STAGE',stage,flush=True)
(ROOT/'pid.json').write_text(json.dumps({'pid':os.getpid(),'stat':Path('/proc/self/stat').read_text(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
status_path=Path('/var/lib/dpkg/status');status_hash=hashlib.sha256(status_path.read_bytes()).hexdigest()
(ROOT/'protected-dpkg-sha256.txt').write_text(status_hash)
try:
    state('SOURCE')
    rev='b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868';archive=ROOT/'source.tar.gz'
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open('https://codeload.github.com/Neroued/ninfer/tar.gz/'+rev,timeout=60) as response,archive.open('xb') as out:
        count=0
        while data:=response.read(1024*1024):
            count+=len(data);assert count<=64*1024**2;out.write(data)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='866f3acbc21f2a9c2cc90fbed49ec4e7cd14b5890126ac0ac3bcca31c2af11f1'
    with tarfile.open(archive,'r:gz') as t:
        for m in t.getmembers():
            parts=PurePosixPath(m.name).parts
            assert parts and parts[0]=='ninfer-'+rev and '..' not in parts and not m.name.startswith('/')
            assert m.isfile() or m.isdir()
        t.extractall(ROOT/'source',filter='data')
    state('PYTHON_TOOLCHAIN')
    requirements=['nvidia-cuda-nvcc==13.1.115','nvidia-cuda-runtime==13.1.80','nvidia-cuda-crt==13.1.115','nvidia-nvvm==13.1.115','nvidia-nvtx==13.1.115','nvidia-cuda-cccl==13.1.115','cmake==3.31.10','ninja==1.13.2']
    (ROOT/'requirements.txt').write_text('\n'.join(requirements)+'\n')
    uv='/home/operator/.local/bin/uv'
    run([uv,'--no-config','--cache-dir',str(ROOT/'uv-cache'),'venv','--python','/usr/bin/python3',str(ROOT/'venv')])
    run([uv,'--no-config','--cache-dir',str(ROOT/'uv-cache'),'pip','install','--python',str(ROOT/'venv/bin/python'),'--only-binary',':all:','--no-deps','--index-url','https://pypi.org/simple','-r',str(ROOT/'requirements.txt')])
    state('APT_DOWNLOAD_ONLY')
    (ROOT/'apt/empty.conf').write_text('')
    shutil.copyfile(status_path,ROOT/'apt/status')
    for p in Path('/var/lib/apt/lists').iterdir():
        if p.is_file() and p.name!='lock':shutil.copyfile(p,ROOT/'apt/lists'/p.name)
    opts={'Dir::State':str(ROOT/'apt'),'Dir::State::status':str(ROOT/'apt/status'),'Dir::State::lists':str(ROOT/'apt/lists'),'Dir::Cache':str(ROOT/'apt'),'Dir::Cache::archives':str(ROOT/'apt/archives'),'Dir::Log':str(ROOT/'apt/log'),'Dir::Etc::main':str(ROOT/'apt/empty.conf'),'Dir::Etc::parts':str(ROOT/'apt/conf.d'),'Dir::Etc::sourcelist':'/dev/null','Dir::Etc::sourceparts':'/etc/apt/sources.list.d'}
    args=['apt-get']
    for k,v in opts.items():args+=['-o',k+'='+v]
    packages=['pkgconf','pkgconf-bin','libpkgconf3','libavcodec-dev','libavformat-dev','libavutil-dev','libswscale-dev','libswresample-dev','libavcodec60','libavformat60','libavutil58','libswscale7','libswresample4','libcurl4-openssl-dev','libcurl4t64']
    args+=['--download-only','--reinstall','--no-install-recommends','--yes','install']+packages
    (ROOT/'apt-command.json').write_text(json.dumps(args,indent=2))
    run(args)
    state('EXTRACT_SYSROOT')
    archives=list((ROOT/'apt/archives').glob('*.deb'));assert archives
    assert sum(p.stat().st_size for p in archives)<1024**3
    for p in archives:run(['dpkg-deb','--extract',str(p),str(ROOT/'sysroot')],timeout=60)
    assert hashlib.sha256(status_path.read_bytes()).hexdigest()==status_hash,'global dpkg status changed'
    state('SETUP_COMPLETE',packages=len(archives),dpkg_status_unchanged=True,source_revision=rev)
except Exception as e:
    state('FAILED',error=repr(e),dpkg_status_unchanged=hashlib.sha256(status_path.read_bytes()).hexdigest()==status_hash)
    raise
