"""Silent two-minute local watchdog: logs, progress, repeat alerts and backups."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

def assess(job,progress,logs,service,now):
    alerts=[]
    if job.get('status')=='failed':alerts.append('PIPELINE_FAILED: '+job.get('error','unknown'))
    if job.get('status')=='running':
        if service not in ['active','activating']:alerts.append('SERVICE_NOT_RUNNING: '+service)
        last=max(progress.get('time',0),job.get('time',job.get('started',0)))
        if now-last>600:alerts.append('NO_PROGRESS_10MIN')
    for name,tail in logs.items():
        if any(x in tail for x in ['Traceback (most recent call last)','CUDA out of memory','Nonfinite','AssertionError','FloatingPointError']):alerts.append('LOG_ERROR: '+name)
    return alerts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--unit',default='research-cseb-student-expanded.service');ap.add_argument('--self-test',action='store_true');a=ap.parse_args();a.evidence.mkdir(parents=True,exist_ok=True)
    if a.self_test:
        errors=assess({'status':'failed','error':'synthetic'}, {}, {'test':'Traceback (most recent call last)'},'failed',time.time())
        stale=assess({'status':'running','started':1},{'time':2},{},'inactive',1000)
        assert len(errors)==2 and len(stale)==2
        p=a.evidence/'monitor-self-test.json';save(p,{'passed':True,'time':time.time(),'delivery':'silent JSON alert file, read back verified','synthetic_alerts':errors+stale});assert json.loads(p.read_text())['passed'];print(p);return
    now=time.time();job=json.loads((a.root/'job.json').read_text())
    progress=json.loads((a.root/'progress.json').read_text()) if (a.root/'progress.json').exists() else {}
    logs={}
    for p in (a.root/'logs').glob('*.log'):
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));logs[p.name]=f.read().decode(errors='replace')
    service=subprocess.run(['systemctl','--user','show',a.unit,'--property=ActiveState','--value'],capture_output=True,text=True,timeout=10).stdout.strip()
    alerts=assess(job,progress,logs,service,now);checkpoints=[]
    for name in ['supervised-best.pt','distilled-best.pt','supervised-last.pt','distilled-last.pt']:
        path=a.root/name
        if not path.exists():continue
        try:
            import torch
            torch.set_num_threads(2);data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
            import io
            ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
            assert all(torch.isfinite(v).all() for v in ck['model'].values())
            assert all(k in ck for k in ['optimizer','epoch','rng','cuda_rng','numpy_rng','python_rng'])
            dest=a.root/'verified-checkpoint-backup'/name;dest.parent.mkdir(exist_ok=True)
            if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:
                tmp=dest.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(dest)
            assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
            checkpoints.append({'name':name,'epoch':ck['epoch'],'sha256':digest,'backup':str(dest),'verified_at':now})
        except Exception as exc:alerts.append('CHECKPOINT_ERROR: '+name+': '+repr(exc))
    previous=json.loads((a.evidence/'alerts.json').read_text()) if (a.evidence/'alerts.json').exists() else {}
    repeat=previous.get('repeat_count',0)+1 if alerts else 0
    status={'time':now,'alerts':alerts,'repeat_count':repeat,'escalated':repeat>=3,'desktop_notifications':False}
    save(a.evidence/'alerts.json',status)
    if alerts:
        with (a.evidence/'alerts.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
    save(a.evidence/'checkpoint-verification.json',{'time':now,'passed':not any(x.startswith('CHECKPOINT_ERROR') for x in alerts),'status':'verified' if checkpoints else 'waiting for first checkpoint','checkpoints':checkpoints})
    save(a.evidence/'heartbeat.json',{'time':now,'job':job,'progress':progress,'service':service,'logs':logs,'alerts':alerts,'coverage':'local systemd timer; silent logs; no automatic agent repair promised'})
    print(json.dumps({'time':now,'phase':job.get('phase'),'alerts':alerts,'checkpoints':len(checkpoints)}))
if __name__=='__main__':main()
