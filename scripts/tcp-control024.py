import socket,errno,json
protected={22,8002,8081,34731,34732}
def fresh():
 s=socket.socket();s.settimeout(3);s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',0));assert s.getsockname()[1] not in protected;return s
def probe(addr,reuse):
 with socket.socket() as s:
  s.settimeout(3)
  if reuse:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind(addr);return 0
  except OSError as e:return e.errno
with fresh() as s:
 addr=s.getsockname();s.listen(1)
 live=probe(addr,True)
 with socket.socket() as client:
  client.settimeout(3);client.connect(addr)
  conn,_=s.accept()
  with conn:
   conn.settimeout(3);conn.sendall(b'x');conn.shutdown(socket.SHUT_WR)
   assert client.recv(1)==b'x';assert client.recv(1)==b''
  # Client remains open until server's active close completed.
raw=probe(addr,False);reuse=probe(addr,True)
with fresh() as s:plain=s.getsockname()
fresh_result=probe(plain,False)
result={'address':addr,'fresh_bind_errno':fresh_result,'live_listener_reuse_errno':live,'after_active_close_raw_errno':raw,'after_active_close_reuse_errno':reuse}
result['passed']=fresh_result==0 and live==errno.EADDRINUSE and raw==errno.EADDRINUSE and reuse==0
print(json.dumps(result));assert result['passed'],result
