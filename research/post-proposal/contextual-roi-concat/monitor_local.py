"""Persistent local alerts. No automatic retries or inference about model quality."""
from pathlib import Path
import json,subprocess,time,os,hashlib,io
import torch
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-concat-20260924')
ROOT.mkdir(exist_ok=True,parents=True)
statefile=ROOT/'monitor-state.json';seen=set(json.loads(statefile.read_text())) if statefile.exists() else set()
def alert(key,message):
    if key in seen:return
    subprocess.run(['notify-send','--urgency=critical','--expire-time=0','Contextual full-context concat monitor',message],timeout=10)
    with (ROOT/'alerts.jsonl').open('a') as f:f.write(json.dumps({'key':key,'message':message,'time':time.time()})+'\n')
    seen.add(key)
if '--test-alert' in __import__('sys').argv:
    alert('monitor-test:'+str(int(time.time())), 'Test: full-context concat monitor alerts are working.');statefile.write_text(json.dumps(sorted(seen)));raise SystemExit(0)
p=ROOT/'pipeline-progress.json';status=json.loads(p.read_text()) if p.exists() else {}
service=subprocess.run(['systemctl','--user','is-active','contextual-roi-concat-study.service'],capture_output=True,text=True).stdout.strip()
if status.get('phase')=='failed' or service=='failed':alert('pipeline-failed:'+str(status.get('time'))+':reminder:'+str(int(time.time()//900)),'Unresolved pipeline failure: '+status.get('error','inspect logs')+'; logs: '+str(ROOT/'logs'))
if status.get('phase') not in ['complete','failed',None] and service!='active':alert('inactive:'+str(status.get('time'))+':reminder:'+str(int(time.time()//900)),'Study service stopped before completion.')
for line in (ROOT/'processes.jsonl').read_text().splitlines() if (ROOT/'processes.jsonl').exists() else []:
    proc=json.loads(line)
    try:
        cmd=Path(f"/proc/{proc['pid']}/cmdline").read_bytes().replace(b'\x00',b' ').decode()
    except (OSError,UnicodeError):continue
    scripts=[arg for arg in proc['command'] if arg.endswith('.py')]
    if not scripts or not all(arg in cmd for arg in scripts):continue
    log=Path(proc.get('log','/nonexistent'))
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-16000));tail=f.read().decode(errors='replace')
        if any(t in tail for t in ['Traceback (most recent call last)', 'CUDA out of memory', 'Nonfinite objective']):alert('log-error:'+str(proc['pid'])+':reminder:'+str(int(time.time()//900)), 'Error in '+str(log))
    latest=max(log.stat().st_mtime if log.exists() else proc['time'], proc['time'])
    if 'compact_context.py' in cmd and p.exists():latest=max(latest,p.stat().st_mtime)
    args=proc['command']
    if '--run' in args:run=args[args.index('--run')+1]
    elif '--seed' in args:
        seed=args[args.index('--seed')+1]
        kind=('contrastive-all184' if float(args[args.index('--contrastive')+1]) else 'classification') if '--contrastive' in args else 'global'
        run='attention-'+kind+'-dcb-seed'+seed
    else:run=None
    own=ROOT/'runs'/run/'progress.json' if run else None
    if own is not None and own.exists():latest=max(latest,own.stat().st_mtime)
    if time.time()-latest>1800:alert('stall:'+str(proc['pid'])+':'+str(int(latest))+':reminder:'+str(int(time.time()//900)),'No study log progress for30 minutes; inspect active processes.')
statefile.write_text(json.dumps(sorted(seen)))
(ROOT/'monitor-heartbeat.json').write_text(json.dumps({'time':time.time(),'service':service,'pipeline':status},indent=2))

verified={};failures={}
for checkpoint in (ROOT/'runs').glob('*/resume.pt'):
    # Atomic writer: bytes correspond to one complete old or new checkpoint.
    try:
        raw=checkpoint.read_bytes()
        ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        assert all(k in ck for k in ['model','optimizer','dcb_state','epoch','position','signature'])
        verified[str(checkpoint)]={'sha256':hashlib.sha256(raw).hexdigest(),'epoch':ck['epoch'],'position':ck['position'],'bytes':len(raw),'verified_at':time.time()}
    except Exception as exc:
        failures[str(checkpoint)]=repr(exc)
        alert('checkpoint:'+str(checkpoint)+':reminder:'+str(int(time.time()//900)), 'Checkpoint verification failed: '+repr(exc))
(ROOT/'checkpoint-verified.json').write_text(json.dumps({'passed':bool(verified) and not failures,'checkpoints':verified,'failures':failures,'time':time.time()},indent=2))
statefile.write_text(json.dumps(sorted(seen)))
