"""CPU-only NCShare watchdog: persistent alerts and heartbeat, no model changes."""
from pathlib import Path
import json,subprocess,time,datetime,hashlib,argparse
ROOT=Path('/work/bbyrd1/research-monitor')
STUDY=Path('/work/bbyrd1/stage56-full-20260914')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(cmd):return subprocess.check_output(cmd,text=True,timeout=45)
def poll():
 ROOT.mkdir(parents=True,exist_ok=True)
 statefile=ROOT/'state.json';state=json.loads(statefile.read_text()) if statefile.exists() else {'notified':[],'first_running':{}}
 alerts=[]
 def alert(key,message,severity='error'):
  if key in state['notified']:return
  item={'id':key,'time':stamp(),'severity':severity,'message':message};alerts.append(item);state['notified'].append(key)
  with (ROOT/'alerts.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
 ids={p.read_text().strip() for p in (STUDY/'results').glob('*-job.txt')};ids={i for i in ids if i.isdigit()}
 queue=run(['squeue','-r','-u','bbyrd1','-h','-o','%i|%j|%T|%M|%R']);live=[]
 for line in queue.splitlines():
  fields=line.split('|',4)
  if len(fields)!=5:continue
  job,name,status,elapsed,reason=fields
  if name.startswith('full56-'):ids.add(job.split('_')[0]);live.append(dict(job=job,name=name,status=status,elapsed=elapsed,reason=reason))
 if ids:
  accounting=run(['sacct','-j',','.join(sorted(ids)),'-X','-P','-n','--format=JobID%30,State,ExitCode'])
  for line in accounting.splitlines():
   f=line.split('|')
   if len(f)<3:continue
   job,status,exitcode=f[:3]
   if any(status.startswith(s) for s in ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL']):alert('failed:'+job,f'NCShare job {job}: {status}, exit {exitcode}. Check experiment logs.')
   if status.startswith('COMPLETED') and '_' in job:alert('complete:'+job,f'NCShare job {job} completed.','info')
 else:accounting=''
 for job in live:
  if job['reason']=='DependencyNeverSatisfied':alert('dependency:'+job['job'],f"Job {job['job']} cannot run because its dependency failed.")
  if job['status']!='RUNNING':continue
  jid=job['job'];state['first_running'].setdefault(jid,time.time());parent,_,index=jid.partition('_');logs=list((STUDY/'results').glob(f'*{parent}-{index if index else "*"}.log'))+list((STUDY/'results').glob(f'*{parent}.log'));latest=max([p.stat().st_mtime for p in logs]+[state['first_running'][jid]])
  if time.time()-latest>1800:alert('stall:'+jid+':'+str(int(latest)),f'Job {jid} has no log progress for over 30 minutes. It may be stalled; inspect before intervening.','warning')
 verified=STUDY/'results/frame-recovery.json'
 if verified.exists() and json.loads(verified.read_text()).get('passed'):
  frames=STUDY/'frames'
  for rel in ['train_00001/00001.jpg','train_00092/00001.jpg']:
   if not (frames/rel).is_file():alert('missing:'+rel,'Required ROAD input disappeared: '+str(frames/rel))
 recovery_log=list((STUDY/'results').glob('frame-recovery-*.log'))
 heartbeat={'checked_utc':stamp(),'checked_unix':time.time(),'jobs':live,'accounting':accounting,'new_alerts':alerts,'recovery_tail':max(recovery_log,key=lambda p:p.stat().st_mtime).read_text()[-1200:] if recovery_log else ''}
 tmp=ROOT/'heartbeat.tmp';tmp.write_text(json.dumps(heartbeat,indent=2));tmp.replace(ROOT/'heartbeat.json');statefile.write_text(json.dumps(state));print(stamp(),len(live),'jobs',len(alerts),'new alerts',flush=True)
 return heartbeat
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
 while True:
  try:poll()
  except Exception as e:
   with (ROOT/'watch-errors.log').open('a') as f:f.write(stamp()+' '+repr(e)+'\n')
   print('WATCH_ERROR',repr(e),flush=True)
  if a.once:break
  time.sleep(120)
