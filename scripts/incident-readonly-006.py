from pathlib import Path
import json,hashlib,subprocess,datetime
r=Path('/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908');b=r/'sglang-baseline';m=json.loads((b/'recovery-manifest.json').read_text())
changes=[]
for p,i in m['protected'].items():
 f=Path(p)
 if not f.exists():changes.append({'path':p,'kind':'missing'});continue
 if 'mtime_ns' in i and (f.stat().st_mtime_ns!=i['mtime_ns'] or f.stat().st_size!=i['bytes']):changes.append({'path':p,'kind':'metadata_changed'})
 if 'sha256' in i and hashlib.sha256(f.read_bytes()).hexdigest()!=i['sha256']:changes.append({'path':p,'kind':'hash_changed'})
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'uptime':Path('/proc/uptime').read_text().strip(),'protected_changes':changes,'checked_files':len(m['protected']),'original_start_ticks':m['original_stat'].split()[21],'original_recovery_utc':m['utc']}
for root,label,name in [(r,'download','download-pid.json'),(b,'server','server-pid.json')]:
 d=json.loads((root/name).read_text());p=Path('/proc')/str(d['pid']);result[label]={'pid':d['pid'],'start_ticks':d['stat'].split()[21],'alive_same_identity':p.exists() and (p/'stat').read_text().split()[21]==d['stat'].split()[21]}
result['ninfer_partial_bytes']=(Path('/home/operator/qwen38-5090/models/NInfer-Normal-11dbbbbb/qwen3_8_27b_nvfp4.ninfer')).stat().st_size
result['download_finished']=(r/'download-result.json').exists()
result['log_tail']=(b/'server.log').read_text().splitlines()[-3:]
result['boots']=subprocess.run(['journalctl','--list-boots','--no-pager'],capture_output=True,text=True,timeout=10).stdout[-3000:]
result['listeners']=subprocess.run(['ss','-ltnp'],capture_output=True,text=True).stdout
print(json.dumps(result,indent=2))
