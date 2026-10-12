from pathlib import Path
import json,shutil
base=Path('/home/operator/qwen38-5090/optimization-runs');template=base/'exp016-common8';r=base/'exp023-kda8';tmp=Path('/tmp/qwen-exp023')
assert not r.exists() and not tmp.exists();r.mkdir();tmp.mkdir();(tmp/'tmp').mkdir()
for n in ['home','storage','crash','downloads','results','profiles']:(r/n).mkdir()
m=json.loads((template/'launch.json').read_text());m['cwd']=str(r);m.pop('wait_for',None)
for flag,value in [('--served-model-name','qwen-normal-exp023'),('--file-storage-path',str(r/'storage')),('--crash-dump-folder',str(r/'crash')),('--download-dir',str(r/'downloads'))]:m['argv'][m['argv'].index(flag)+1]=value
m['environment'].update(HOME=str(r/'home'),HF_HOME=str(r/'home/huggingface'),XDG_CACHE_HOME=str(r/'home/.cache'),TMPDIR=str(tmp/'tmp'))
(r/'patch').mkdir();(r/'cute-cache').mkdir();(r/'cute-dumps').mkdir();(r/'cuda-cache').mkdir()
(r/'patch/sitecustomize.py').write_bytes(__import__('base64').b64decode('IiIiRVhQMDIzIHByb2Nlc3MtbG9jYWwgZWxpZ2liaWxpdHkgZXh0ZW5zaW9uOyBpbnN0YWxsZWQgc291cmNlIHJlbWFpbnMgdW5jaGFuZ2VkLiIiIgppbXBvcnQgaGFzaGxpYgppbXBvcnQgb3MKaW1wb3J0IHN5cwpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCnRyeToKICAgIGltcG9ydCBzZ2xhbmcua2VybmVscy5rZGFfa2VybmVscy5xd2VuM3hfbnZmcDRfZ2VtbSBhcyBnYXRlCiAgICBleHBlY3RlZCA9ICdiMTRjM2VhYTZiOGQ5ZDYyYjU1OWE1MGJlNjJmZTRjYWQ5MzVlNzE3MmJkYjc5ZDE4YTFmNmFiY2I0Y2NiN2M1JwogICAgaWYgaGFzaGxpYi5zaGEyNTYoUGF0aChnYXRlLl9fZmlsZV9fKS5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpICE9IGV4cGVjdGVkOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignRVhQMDIzIGdhdGUgc291cmNlIGhhc2ggbWlzbWF0Y2gnKQogICAgaWYgKDgsIDE3NDA4LCA1MTIwKSBpbiBnYXRlLl9TVVBQT1JURURfU0hBUEVTOgogICAgICAgIHJhaXNlIFJ1bnRpbWVFcnJvcignRVhQMDIzIGJhc2VsaW5lIGVsaWdpYmlsaXR5IGNoYW5nZWQnKQogICAgZ2F0ZS5fU1VQUE9SVEVEX1NIQVBFUyA9IGdhdGUuX1NVUFBPUlRFRF9TSEFQRVMudW5pb24oeyg4LCAxNzQwOCwgNTEyMCl9KQogICAgcHJpbnQoJ0VYUDAyM19LREFfTThfRU5BQkxFRCBwaWQ9JyArIHN0cihvcy5nZXRwaWQoKSksIGZpbGU9c3lzLnN0ZGVyciwgZmx1c2g9VHJ1ZSkKZXhjZXB0IEV4Y2VwdGlvbiBhcyBleGM6CiAgICBwcmludCgnRVhQMDIzX1BBVENIX0ZBSUxFRDogJyArIHJlcHIoZXhjKSwgZmlsZT1zeXMuc3RkZXJyLCBmbHVzaD1UcnVlKQogICAgIyBQeXRob24gbm9ybWFsbHkgaWdub3JlcyBzaXRlY3VzdG9taXplIGVycm9yczsgbmV2ZXIgcnVuIGFuIHVucGF0Y2hlZCB0cmlhbC4KICAgIG9zLl9leGl0KDc4KQo='))
m['environment'].update(PYTHONPATH=str(r/'patch'),CUTE_DSL_CACHE_DIR=str(r/'cute-cache'),CUTE_DSL_DUMP_DIR=str(r/'cute-dumps'),CUDA_CACHE_PATH=str(r/'cuda-cache'))
(r/'launch.json').write_text(json.dumps(m,indent=2))
for name in ['exec-manifest.py','benchmark_stream.py']:shutil.copyfile(template/name,r/name)
shutil.copytree(template/'cases',r/'cases');print('PREPARED023')
