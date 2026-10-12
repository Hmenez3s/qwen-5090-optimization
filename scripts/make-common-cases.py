from pathlib import Path
import re,json,hashlib
from tokenizers import Tokenizer
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/common-cases')
assert not ROOT.exists();ROOT.mkdir()
SRC=Path('/home/operator/qwen38-5090/optimization-runs/exp010-context/cases')
tok=Tokenizer.from_file('/home/operator/qwen38-5090/models/Qwen3.8-27B-NVFP4-LMHead4/tokenizer.json')
meta=[]
for name in ['performance','quality-needles']:
 d=json.loads((SRC/(name+'.json')).read_text());cases=[]
 for case in d['cases']:
  ids=case['payload']['input_ids'];text=tok.decode(ids,skip_special_tokens=False)
  matches=list(re.finditer(r'<\|im_start\|>(system|user)\n(.*?)<\|im_end\|>',text,re.S))
  assert matches and text[matches[-1].end():].lstrip().startswith('<|im_start|>assistant')
  messages=[{'role':m.group(1),'content':m.group(2)} for m in matches]
  cases.append({'name':case['name'],'payload':{'messages':messages,'temperature':0,'top_p':1,'max_tokens':1024}})
  meta.append({'case':case['name'],'native_input_tokens':len(ids),'decoded_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'note':'same semantic messages; each server applies own template; compare actual reported prompt counts'})
 with (ROOT/(name+'.json')).open('x') as f:json.dump({'cases':cases,'repetitions':d['repetitions']},f,separators=(',',':'))
(ROOT/'metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({'root':str(ROOT),'cases':len(meta)}))
