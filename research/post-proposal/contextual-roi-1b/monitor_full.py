"""Persistent two-minute status checks and hourly verified off-cluster checkpoints.

No automatic scientific changes or repeated desktop notifications. Failures remain
in alerts.json and monitor-error.json for the repair operator/active session.
"""
from pathlib import Path
import json,subprocess,time,hashlib,os

OUT=Path('/data/repos/wiki/artifacts/contextual-roi-1b-20260930/monitor')
BACKUP=Path('/home/brandon/.local/share/road-reason/contextual-roi-1b-20260930')
REMOTE='/work/bbyrd1/contextual-roi-1b-20260930'
SCRIPT=r'''
from pathlib import Path
import json,subprocess,time,os
r=Path('/work/bbyrd1/contextual-roi-1b-20260930');now=time.time();alerts=[];records={}
launch=r/'results/full-launch.json';jobs=json.loads(launch.read_text())['jobs'] if launch.exists() else {}
queue=subprocess.check_output(['squeue','-r','-u','bbyrd1','-h','-o','%i|%j|%T|%R'],text=True,timeout=30)
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs.values()),'--format=JobID%30,State,ExitCode'],text=True,timeout=30) if jobs else ''
for line in accounting.splitlines():
 if any(s in line for s in ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','CANCELLED']):alerts.append(line)
for line in queue.splitlines():
 if 'full56-1b-' in line and 'DependencyNeverSatisfied' in line:alerts.append(line)
for pattern in ['results/*.json','runs/*/progress.json','runs/*/epochs.json','runs/*/complete.json','runs/*/detector-results.json']:
 for p in r.glob(pattern):
  if p.name in ['frame-hashes.json']:continue
  records[str(p.relative_to(r))]=json.loads(p.read_text())
logs={}
for p in r.glob('results/full56-1b-*.log'):
 if not any('-'+job+'-' in p.name for job in jobs.values()):continue
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
 logs[p.name]={'mtime':p.stat().st_mtime,'tail':tail}
 if 'Traceback (most recent call last)' in tail or 'CUDA out of memory' in tail:alerts.append('LOG_ERROR '+p.name)
for line in queue.splitlines():
 fields=line.split('|');jid=fields[0]
 if len(fields)<3 or fields[2]!='RUNNING' or 'full56-1b-' not in fields[1]:continue
 parent,_,index=jid.partition('_');matches=[v for k,v in logs.items() if '-'+parent+'-'+(index or '4294967294')+'.log' in k]
 if matches and now-max(v['mtime'] for v in matches)>1800:alerts.append('NO_LOG_PROGRESS_30MIN '+jid)
backup=r/'checkpoint-backup';backup.mkdir(exist_ok=True);manifest=backup/'snapshot.json'
due=not manifest.exists() or now-manifest.stat().st_mtime>3500
if due:
 files=[]
 for run in sorted((r/'runs').glob('*')):
  for name in ['resume.pt','best.pt']:
   src=run/name
   if not src.exists():continue
   dest=backup/run.name/name;dest.parent.mkdir(exist_ok=True);tmp=dest.with_suffix('.link');tmp.unlink(missing_ok=True);os.link(src,tmp);tmp.replace(dest)
   files.append({'path':str(dest.relative_to(backup)),'bytes':dest.stat().st_size})
 tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps({'time':now,'files':files},indent=2));tmp.replace(manifest)
print(json.dumps({'time':now,'root':str(r),'jobs':jobs,'queue':queue,'accounting':accounting,'records':records,'logs':logs,'alerts':alerts,'snapshot':json.loads(manifest.read_text())}))
'''

def atomic(d,p):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True);BACKUP.mkdir(parents=True,exist_ok=True)
    response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare','python3 -'],input=SCRIPT,text=True,capture_output=True,check=True,timeout=150)
    d=json.loads(response.stdout);atomic(d,OUT/'heartbeat.json');atomic({'time':time.time(),'alerts':d['alerts']},OUT/'alerts.json')
    verified=OUT/'backup-verified.json';last=json.loads(verified.read_text()) if verified.exists() else {}
    if d['snapshot']['time']>last.get('snapshot_time',0):
        subprocess.run(['rsync','-rt','--delay-updates','--timeout=120','--include=*/','--include=*.pt','--include=snapshot.json','--exclude=*','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','ncshare:'+REMOTE+'/checkpoint-backup/',str(BACKUP)+'/'],check=True,timeout=1800)
        import torch
        torch.set_num_threads(4);checked=[]
        for item in d['snapshot']['files']:
            p=BACKUP/item['path'];assert p.stat().st_size==item['bytes'];ck=torch.load(p,map_location='cpu',weights_only=False)
            if p.name=='resume.pt':assert all(k in ck for k in ['model','optimizer','dcb_state','torch_rng','cuda_rng','python_rng','numpy_rng','steps','position','epoch'])
            states=ck['models'] if ck.get('kind')=='blend' else [ck['model']]
            assert all(torch.isfinite(v).all() for state in states for v in state.values())
            checked.append({'path':item['path'],'bytes':item['bytes'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        atomic({'passed':True,'time':time.time(),'snapshot_time':d['snapshot']['time'],'checkpoints':checked,'status':'verified' if checked else 'waiting for first training checkpoint'},verified)
    summary=d['records'].get('results/full-summary.json')
    if summary and summary.get('passed') and not (OUT/'completion-ping.json').exists():
        atomic({'time':time.time(),'attempted':True},OUT/'completion-ping.json')
        sent=subprocess.run(['notify-send','InternVideo2-1B study complete','All three seeds, expert blends and nine detector evaluations are complete.'],timeout=10).returncode==0
        if sent:atomic({'time':time.time(),'delivered':True},OUT/'completion-ping.json')
    (OUT/'monitor-error.json').unlink(missing_ok=True)
    print(json.dumps({'time':time.time(),'jobs':d['jobs'],'alerts':d['alerts'],'backup':str(verified)}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);atomic({'time':time.time(),'error':repr(exc)},OUT/'monitor-error.json');raise
