"""Two-minute per-process progress checks, persistent repeated alerts."""
from pathlib import Path
import json,subprocess,time,os
ROOT=Path('/data/repos/wiki/artifacts/contextual-gate-20260919');now=time.time()
def read(p,default):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def alert(key,message):
 p=ROOT/'monitor-state.json';seen=read(p,{});key=key+':reminder:'+str(int(now//900))
 if key in seen:return
 subprocess.run(['notify-send','--urgency=critical','Contextual gate monitor',message],timeout=10)
 with (ROOT/'alerts.jsonl').open('a') as f:f.write(json.dumps({'key':key,'message':message,'time':now})+'\n')
 seen[key]=now;p.write_text(json.dumps(seen))
state=read(ROOT/'pipeline-progress.json',{});service=subprocess.run(['systemctl','--user','is-active','contextual-gate-study.service'],capture_output=True,text=True).stdout.strip()
if state.get('phase')=='failed' or service=='failed':alert('gate-failed:'+str(state.get('time')),state.get('error','Study service failed'))
elif state.get('phase') not in [None,'complete'] and service!='active':alert('gate-inactive','Gate study stopped before completion')
for line in (ROOT/'processes.jsonl').read_text().splitlines() if (ROOT/'processes.jsonl').exists() else []:
 proc=json.loads(line)
 try:cmd=Path(f"/proc/{proc['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
 except (OSError,UnicodeError):continue
 args=proc['command'];scripts=[a for a in args if a.endswith('.py')]
 if not scripts or not all(a in cmd for a in scripts):continue
 paths=[Path(proc['log'])]
 if '--seed' in args:paths.append(ROOT/'runs'/('seed'+args[args.index('--seed')+1])/'progress.json')
 if '--shard' in args:paths.append(ROOT/'context'/('progress-'+args[args.index('--shard')+1]+'.json'))
 latest=max([proc['time']]+[p.stat().st_mtime for p in paths if p.exists()])
 if now-latest>1800:alert('stall:'+str(proc['pid']),f'No progress30min: {proc["log"]}')
if state.get('phase') not in [None,'complete','failed']:
 import shutil
 if shutil.disk_usage(ROOT).free<5*1024**3:alert('disk-low','Gate study data disk below5GiB free; inspect without deleting research data')
(ROOT/'monitor-heartbeat.json').write_text(json.dumps({'time':now,'service':service,'pipeline':state},indent=2))
