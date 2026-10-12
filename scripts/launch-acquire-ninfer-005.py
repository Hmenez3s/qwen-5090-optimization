from pathlib import Path
import json,subprocess,os,base64,zlib
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908')
MANIFEST=json.loads(zlib.decompress(base64.b64decode('eJx1kt2O0zAQRl9l1ettY8/4d+8WqMRK7EpQcYVQZXvGbVCTlCRdJFZ9d5wFpIaKXFmxfc545ntZEA9j3Yax7trF3c2i2ncNV8O+r5tj9f0Ht+iWWnhRNR3xYaieHtrM/fKp65twWEpJcfoWtzeLGAbenvrDRNmP43G4q6r9aber210OiVepq1ruuxNT9XHirtwS7Jtl+5yPavkbW/U8dIdnrv5yEyJF6V3IDFlA8oCQrJU5W4Ey8uTN9YGHIv3ysuindRsanmpY7eoxjGNfx9NYDkwVHrr4QNOeZS2RjPOSsvdCg8rKgaMQQCoSWagQNBo33RrqnxNPaifOtzf/SD48vF0/bdZzOhjJHD377CJEFQVJ6dEYouB9MpksoU5GXdIlanuN/7S+f/e4XjU0FwTnGRRJa11hlnYUKDgQEYsnBY4ZNEUdZuUL4a4Fm/f3oM3m8+NmbtCkAyvjCDRnAu8ducRRhuSAmL32umwJf2Hw8hof+rEuwx+XTWjrXJK2+jaUnM1U5FAaK0nrkCHarBgNoORsKZAAxY5seQ1cqAARr2Wvad26Ldi4fU3Vqq2nVM113pbBRCml5Zw0AASnSgoilWEhI7oglDIoZzorvfJGeig/D3lK28ti2IfSutdmaUhoVTJOU2IUYARGr3Nkr0SOQoHCJBIplYTDojVKBULyTpAykP6rOnZ1O3K/+TNCNOfz1/MvcrYUig==')))
DEST=Path(MANIFEST['destination'])
assert not ROOT.exists(), 'run exists'
assert not DEST.exists(), 'model exists'
assert os.statvfs('/home/operator').f_bavail*os.statvfs('/home/operator').f_frsize>40*1024**3
ROOT.mkdir(parents=True);DEST.mkdir()
(ROOT/'download-manifest.json').write_text(json.dumps(MANIFEST,indent=2))
(ROOT/'download.py').write_bytes(zlib.decompress(base64.b64decode('eJyNVE1v2zgQvetXsCdKG1lW3E2QVaECRdcFjLZJkPX2EhgGLVERG4lU+QG7DfLfd4aS7CRugNVB0gzfDB9nHke0ndKWFEoWTmsubVI56zQ3MamZqRuxicl3o2RMrGh5TJxuwJdo/sNxY4NKq5Z0zCKQiD7XNZjBzdXVMse/kE5r1fKpqTWsT39suXx7MTlL/0qnqoOc4hezQsmJdtJMpdItayZ3Cl7VJj07L9KzySydnQP8gkZBy6SoYN8cKSWNYqUJQ9xqSku1leiYjJgEMTQCqqxcW76zYRQFJQZ7WiPslqJPSM+CrqLAWKYtL3M8b9IqqaySoggxtiIVt0UdCsvbKAsIPJK1PEf7lupKNBxtuvJLoiJ0SomQHkSU9t/E5zdbgaVJ6JAGH82h8JI8UMxDMwTHFNDWGZrRfz4vrteXV5frm38vl4uvc/roA7sc6U8R7G3oT3442oYZvgYXXZ0AlZM9yohfPE/fkVLcYUGGTiemZrOzczgqYpAh6RLVcRnS3YZGhBminD0Q9ojngkjA9BHwjbGCEJCfzlIfDKo6BPsENZyULLXjz/1D9V52YDK05v1pepGmmWbCQHS/yVxrpUNqlWUN2bjyjlvCdwXnJS9pdJS9qJ28z4GQl0f45x+n6ax/HWOBCVDoQ7IN4O+PIEAg2WqQQehR0Tss8EkOchgdfaUT15Vsj/Jp+K7jBeoNRZQA7ZA2laHxw2PUW31ThjMAF0xN3uSkFx1adDXU4htrHB8q4WGtMC0DxZK+9/sc46aEyXIQQVLzXf8XRm/yEfCbxCiW3yfWapu/ot5v85vFp8X8bxrTzU/LwYX84vFw2TGHmI4c1iNodAzK10La0A+C0rWdCWH7KK4aZ+ocJTVw6i8VrAVW/8wOyj6eeMmyRjFcK9XMd7xwFk7bst16q/Q91yafeRF3sPzkzqqtyRsBhNGftKwL/YiID3cQ62FgsIx8jGsslGlfmo9XX6+/zJdzKE0PzTAplIYDxRLM167BY4AC7yyZ+w+ML+QHvuyVnT59WHzxLeDYSdiHdzoEfPS/Nns5Z/v845T16u/H7NOWeEwsZAl1hgJGwVHXDvx68O1or56w2i8NjtXj807/B8QQOjY=')))
with (ROOT/'download.log').open('xb') as log:
 p=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'download.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={'PATH':'/usr/bin:/bin','HOME':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'})
(ROOT/'download-pid.json').write_text(json.dumps({'pid':p.pid,'stat':Path('/proc',str(p.pid),'stat').read_text()}))
print(json.dumps({'pid':p.pid,'destination':str(DEST),'run':str(ROOT)}))
