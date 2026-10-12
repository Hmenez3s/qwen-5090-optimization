from pathlib import Path
import sys,json,urllib.request,datetime
ROOT=Path('/home/operator/qwen38-5090/optimization-runs/exp015b-ninfer8')
sys.path.insert(0,str(ROOT))
from benchmark_stream import build_isolated_opener
opener=build_isolated_opener();base='http://127.0.0.1:34732';model='qwen-normal-exp015b'
with opener.open(base+'/v1/models',timeout=10) as r:assert model in [x['id'] for x in json.load(r)['data']]
out=ROOT/'results/quality-api.jsonl'
def ask(messages,tools=None):
 body={'model':model,'messages':messages,'temperature':0,'max_tokens':4096}
 if tools:body['tools']=tools
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with opener.open(req,timeout=180) as r:return json.load(r)
def objects(text):
 out=[]
 for i,ch in enumerate(text or ''):
  if ch=='{':
   try:out.append(json.JSONDecoder().raw_decode(text[i:])[0])
   except json.JSONDecodeError:pass
 return out
with out.open('x') as log:
 reasoning_seen=False
 for name,prompt,expected in [
  ('product','Compute 987 times 123. Check your arithmetic. Return the final answer as JSON with key product.',{'product':121401}),
  ('gcd','Use the Euclidean algorithm to find gcd(1071,462). Return final JSON with key gcd.',{'gcd':21}),
  ('probability','A bag has5 red and3 blue balls. Draw2 without replacement. What is probability both red? Return final JSON with key probability and a reduced fraction string.',{'probability':'5/14'}),
  ('crt','Find the smallest nonnegative integer x with x mod5=2 and x mod7=3. Return final JSON with key x.',{'x':17})]:
  response=ask([{'role':'user','content':prompt}]);choice=response['choices'][0];message=choice['message']
  reasoning=message.get('reasoning_content') or message.get('reasoning') or ''
  reasoning_seen=reasoning_seen or bool(reasoning)
  passed=expected in objects(message.get('content')) and choice['finish_reason']=='stop'
  log.write(json.dumps({'name':name,'passed':passed,'reasoning_chars':len(reasoning),'response':response})+'\n');log.flush()
  print(name,passed,'reasoning_chars',len(reasoning),flush=True)
  assert passed,name
 tools=[{'type':'function','function':{'name':'add_numbers','description':'Add two integers and return their sum.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}]
 messages=[{'role':'user','content':'Use add_numbers to calculate17+25. After receiving its result, return final JSON with key sum.'}]
 first=ask(messages,tools);m=first['choices'][0]['message'];calls=m.get('tool_calls') or []
 passed=len(calls)==1 and calls[0]['function']['name']=='add_numbers' and json.loads(calls[0]['function']['arguments'])=={'a':17,'b':25}
 log.write(json.dumps({'name':'tool-emission','passed':passed,'response':first})+'\n');log.flush();assert passed,'tool-emission'
 assistant={'role':'assistant','content':m.get('content'),'tool_calls':calls}
 for k in ['reasoning_content','reasoning']:
  if m.get(k):assistant[k]=m[k]
 messages += [assistant,{'role':'tool','tool_call_id':calls[0]['id'],'content':json.dumps({'result':42})}]
 final=ask(messages,tools);c=final['choices'][0]
 passed={'sum':42} in objects(c['message'].get('content')) and c['finish_reason']=='stop'
 log.write(json.dumps({'name':'tool-roundtrip','passed':passed,'response':final})+'\n');log.flush();print('tool-roundtrip',passed,flush=True)
 assert passed,'tool-roundtrip'
 assert reasoning_seen,'reasoning trace not observed'
 print('ALL_FOCUSED_CHECKS_PASS',flush=True)
