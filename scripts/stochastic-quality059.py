"""Three-seed stochastic semantic gate for the final KDA2 service."""
from pathlib import Path
import json,urllib.error,urllib.request
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/pi-serving-059');OUT=ROOT/'results/stochastic-quality.jsonl';MODEL='qwen3.8-27b-normal'
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):raise urllib.error.HTTPError(args[0].full_url,args[2],'redirects disabled',args[3],args[4])
op=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
o=json.loads((ROOT/'server-pid.json').read_text());p=Path('/proc')/str(o['pid']);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==o['boot_id'] and (p/'stat').read_text().split()[21]==o['stat'].split()[21]
info=json.loads(op.open('http://127.0.0.1:8081/get_server_info',timeout=10).read());assert info['status']=='ready' and info['served_model_name']==MODEL and info['context_length']==131200 and info['max_total_num_tokens']==131328 and info['speculative_dflash_block_size']==8
assert not OUT.exists() and OUT.parent.is_dir()
def objects(text):
 out=[]
 for i,ch in enumerate(text or ''):
  if ch=='{':
   try:out.append(json.JSONDecoder().raw_decode(text[i:])[0])
   except json.JSONDecodeError:pass
 return out
with OUT.open('x') as log:
 for seed in (17,29,43):
  body={'model':MODEL,'messages':[{'role':'user','content':'Compute 987 times 123. Check the arithmetic. Return final JSON with key product.'}],'temperature':0.7,'top_p':1,'top_k':50,'min_p':0,'seed':seed,'max_tokens':4096,'chat_template_kwargs':{'enable_thinking':True}}
  req=urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with op.open(req,timeout=180) as response:d=json.load(response)
  choice=d['choices'][0];message=choice['message'];reasoning=message.get('reasoning_content') or message.get('reasoning') or '';passed={'product':121401} in objects(message.get('content')) and choice.get('finish_reason')=='stop' and bool(reasoning)
  log.write(json.dumps({'seed':seed,'passed':passed,'reasoning_chars':len(reasoning),'response':d})+'\n');log.flush();assert passed,seed
 log.write(json.dumps({'record_type':'summary','passed':True,'seeds':[17,29,43],'claim_scope':'synthetic stochastic semantic quality only'})+'\n')
print(json.dumps({'status':'PASS','seeds':3,'output':str(OUT)}))
