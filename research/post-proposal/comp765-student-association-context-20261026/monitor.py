"""Silent persistent progress/artifact monitor; no arbitrary repairs promised."""
from assoc_common import *
import argparse,shutil,subprocess
UNIT='research-comp765-student-association-context.service'
def assess(job,prog,logs,state,now):
 alerts=[]
 if job.get('status')=='failed':alerts.append('PIPELINE_FAILED: '+job.get('error','unknown'))
 if job.get('status')=='running':
  if state not in ['active','activating']:alerts.append('SERVICE_NOT_RUNNING: '+state)
  if now-max(job.get('time',0),prog.get('time',0))>600:alerts.append('NO_PROGRESS_10MIN')
 for name,tail in logs.items():
  if any(s in tail for s in ['Traceback (most recent call last)','CUDA out of memory','AssertionError','FloatingPointError']):alerts.append('LOG_ERROR: '+name)
 return alerts

def main(test=False):
 evidence=ROOT/'monitor';evidence.mkdir(exist_ok=True);now=time.time()
 if test:
  a=assess({'status':'failed','error':'synthetic'},{},{'test':'Traceback (most recent call last)'},'failed',now);b=assess({'status':'running','time':1},{'time':2},{},'inactive',1000);assert len(a)==2 and len(b)==2
  save(evidence/'self-test.json',dict(passed=True,time=now,synthetic_alerts=a+b,delivery='Silent JSON logs; readback verified; no banner'));assert read(evidence/'self-test.json')['passed'];return
 job=read(ROOT/'job.json') if (ROOT/'job.json').exists() else {'status':'prelaunch'};prog=read(ROOT/'progress.json') if (ROOT/'progress.json').exists() else {};logs={}
 for name in ['pipeline','mine','preflight','official-benchmark','official','runtime','replay_audit','analyze','audit']:
  path=ROOT/'logs'/f'{name}.log'
  if path.exists():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-6000));logs[name]=f.read().decode(errors='replace')
 state=subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','--value'],capture_output=True,text=True,timeout=10).stdout.strip();alerts=assess(job,prog,logs,state,now);verified=[]
 if shutil.disk_usage(ROOT).free<1*1024**3:alerts.append('DISK_FREE_BELOW_1GiB')
 for seed in SEEDS:
  path=checkpoint(seed);backup=BASE_ROOT/'verified-checkpoint-backup'/path.parent.name/'best.pt'
  try:
   h=sha(path);assert h==read(path.parent/'training-complete.json')['checkpoint_sha256']==sha(backup);verified.append(dict(path=str(path),sha256=h,backup=str(backup),verified_at=now))
  except Exception as e:alerts.append('CHECKPOINT_ERROR: '+repr(e))
 if (ROOT/'freeze.json').exists():
  try:verify()
  except Exception as e:alerts.append('FROZEN_ASSET_ERROR: '+repr(e))
 prior=read(evidence/'alerts.json') if (evidence/'alerts.json').exists() else {};repeat=prior.get('repeat_count',0)+1 if alerts else 0;status=dict(time=now,alerts=alerts,repeat_count=repeat,escalated=repeat>=3,desktop_notifications=False)
 save(evidence/'alerts.json',status)
 if alerts:
  with (evidence/'alerts.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
 save(evidence/'checkpoints.json',dict(time=now,passed=len(verified)==3,checkpoints=verified));save(evidence/'heartbeat.json',dict(time=now,job=job,progress=prog,state=state,alerts=alerts,logs=logs,coverage='2min timer; logs/stalls/frozen hashes/original checkpoint backups; no arbitrary repair or conversational wakeup guaranteed'));print(json.dumps(status))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args();main(a.self_test)
