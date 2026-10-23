"""Persistent silent monitoring, repeated alerts, verified checkpoint backups."""
import argparse, io, shutil, subprocess, time
from common import *

UNIT='research-comp765-student-efficientnet.service'
def assess(job,progress,logs,state,now):
    alerts=[]
    if job.get('status')=='failed':alerts.append('PIPELINE_FAILED: '+job.get('error','unknown'))
    if job.get('status')=='running':
        if state not in ['active','activating']:alerts.append('SERVICE_NOT_RUNNING: '+state)
        if now-max(job.get('time',0),progress.get('time',0))>600:alerts.append('NO_PROGRESS_10MIN')
    for name,tail in logs.items():
        if any(s in tail for s in ['Traceback (most recent call last)','CUDA out of memory','Nonfinite','AssertionError','FloatingPointError']):alerts.append('LOG_ERROR: '+name)
    return alerts

def main(root,self_test=False):
    evidence=root/'monitor';evidence.mkdir(exist_ok=True);now=time.time()
    if self_test:
        a=assess({'status':'failed','error':'synthetic'},{},{'test':'Traceback (most recent call last)'},'failed',now)
        b=assess({'status':'running','time':1},{'time':2},{},'inactive',1000)
        assert len(a)==2 and len(b)==2
        save(evidence/'self-test.json',dict(passed=True,time=now,delivery='Silent JSON logs; read back verified',synthetic_alerts=a+b));assert read(evidence/'self-test.json')['passed'];return
    job=read(root/'job.json') if (root/'job.json').exists() else {'status':'prelaunch'}
    prog=read(root/'progress.json') if (root/'progress.json').exists() else {};logs={}
    for name in ['pipeline','prepare','train','evaluate','runtime_audit','analyze']:
        p=root/'logs'/f'{name}.log'
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));logs[name]=f.read().decode(errors='replace')
    state=subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','--value'],capture_output=True,text=True,timeout=10).stdout.strip()
    alerts=assess(job,prog,logs,state,now);verified=[]
    if shutil.disk_usage(root).free<2*1024**3:alerts.append('DISK_FREE_BELOW_2GiB')
    previous=read(evidence/'checkpoints.json') if (evidence/'checkpoints.json').exists() else {'checkpoints':[]}
    known={c['path']:c for c in previous['checkpoints']}
    for p in (root/'runs').glob('*/*.pt'):
        try:
            data=p.read_bytes();h=hashlib.sha256(data).hexdigest();backup=root/'verified-checkpoint-backup'/p.parent.name/p.name
            if str(p) in known and known[str(p)]['sha256']==h and backup.exists() and sha(backup)==h:
                entry=known[str(p)];entry['verified_at']=now;verified.append(entry);continue
            ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
            assert all(torch.isfinite(v).all() for v in ck['model'].values())
            if p.name=='last.pt':assert all(k in ck for k in ['optimizer','rng','cuda_rng','numpy_rng','python_rng','offset'])
            backup.parent.mkdir(parents=True,exist_ok=True);temp=backup.with_suffix('.tmp');temp.write_bytes(data);temp.replace(backup);assert sha(backup)==h
            verified.append(dict(path=str(p),sha256=h,epoch=ck['epoch'],backup=str(backup),verified_at=now))
        except Exception as e:alerts.append('CHECKPOINT_ERROR: '+str(p)+': '+repr(e))
    prior=read(evidence/'alerts.json') if (evidence/'alerts.json').exists() else {};repeat=prior.get('repeat_count',0)+1 if alerts else 0
    status=dict(time=now,alerts=alerts,repeat_count=repeat,escalated=repeat>=3,desktop_notifications=False)
    save(evidence/'alerts.json',status)
    if alerts:
        with (evidence/'alerts.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
    save(evidence/'checkpoints.json',dict(time=now,passed=not any(a.startswith('CHECKPOINT_ERROR') for a in alerts),checkpoints=verified))
    save(evidence/'heartbeat.json',dict(time=now,job=job,progress=prog,state=state,logs=logs,alerts=alerts,coverage='2min systemd timer; logs, stalls, space and verified backups; no automatic conversational wakeup or arbitrary repair promised'))
    print(json.dumps(dict(time=now,alerts=alerts,verified=len(verified))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--self-test',action='store_true');a=ap.parse_args();torch.set_num_threads(2);main(a.root,a.self_test)
