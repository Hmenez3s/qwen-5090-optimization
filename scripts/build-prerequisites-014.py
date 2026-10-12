from pathlib import Path
import subprocess,urllib.parse
site=Path('/home/operator/.venvs/qwen38/lib/python3.12/site-packages')
print('NVIDIA_PACKAGES',[p.name for p in site.glob('nvidia*dist-info')])
for p in Path('/etc/apt/sources.list.d').glob('*.sources'):
 for line in p.read_text().splitlines():
  if line.startswith('URIs:'):
   for u in line[5:].split():
    x=urllib.parse.urlsplit(u);assert x.username is None and x.password is None
    print('APT_HOST',x.hostname)
print(subprocess.run(['apt-cache','policy','pkgconf','pkgconf-bin','libpkgconf3','libavcodec-dev','libavformat-dev','libavutil-dev','libswscale-dev','libcurl4-openssl-dev'],capture_output=True,text=True).stdout[:9000])
