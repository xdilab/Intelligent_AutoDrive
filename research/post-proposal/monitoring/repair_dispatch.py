"""Two-minute local incident dispatcher; serialized, bounded Codex repair sessions."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,time
ROOT=Path('/data/repos/wiki/artifacts/research-repair');CODE=Path(__file__).parent
WIKI=Path('/data/repos/wiki');CTX=WIKI/'artifacts/contextual-roi-20260917'
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def notify(s):
 try:subprocess.run(['notify-send','--urgency=critical','Research repair operator',s],timeout=10)
 except (OSError,subprocess.TimeoutExpired):pass

def resolved_accounting(line):
 """Only retire the exact terminal record backed by a verified completed report."""
 entry=read(ROOT/'resolved-incidents.json',{}).get('cluster-job:'+line.split('|')[0],{})
 if entry.get('accounting')!=line:return False
 try:
  evidence=Path(entry['evidence']);raw=evidence.read_bytes();report=json.loads(raw)
  return (hashlib.sha256(raw).hexdigest()==entry['sha256'] and report.get('passed') is True
          and len(report.get('epochs',[]))==report['protocol']['epochs']
          and report.get('completed_utc')==entry['completed_utc'])
 except (OSError,ValueError,KeyError,TypeError):return False

def continued_accounting(line,live_jobs):
 """A timed-out parent is healthy while its registered continuation is live."""
 parent=line.split('|')[0]
 continuations=read(ROOT/'active-continuations.json',{})
 child=continuations.get(parent)
 if not child:return False
 return any(str(j.get('job'))==str(child) and j.get('status') in ('RUNNING','PENDING')
            for j in live_jobs)

def collect(now):
 issues=[]
 def add(key,detail):issues.append({'key':key,'detail':detail})
 p=read(CTX/'pipeline-progress.json',{});h=read(CTX/'monitor-heartbeat.json',{})
 if p.get('phase')=='failed':add('context-failed:'+str(p.get('time')),p)
 elif p.get('phase') not in ('complete',None):
  if now-h.get('time',0)>600:add('local-monitor-stale',h)
  elif h.get('service')!='active':add('context-inactive:'+str(p.get('time')),h)
  # Reuse live-process-aware monitor's alerts; recent repeat keys keep active stalls visible.
  for line in (CTX/'alerts.jsonl').read_text().splitlines()[-100:] if (CTX/'alerts.jsonl').exists() else []:
   a=json.loads(line)
   if now-a.get('time',0)<600 and 'stall:' in a.get('key',''):add(a['key'].split(':reminder:')[0],a)
 extra=WIKI/'artifacts/contextual-roi-all184-20260918'
 ep=read(extra/'pipeline-progress.json',{});eh=read(extra/'monitor-heartbeat.json',{})
 if ep.get('phase')=='failed':add('all184-failed:'+str(ep.get('time')),ep)
 elif ep.get('phase') not in ('complete',None):
  if now-eh.get('time',0)>600:add('all184-monitor-stale',eh)
  elif eh.get('service')!='active':add('all184-inactive:'+str(ep.get('time')),eh)
  for line in (extra/'alerts.jsonl').read_text().splitlines()[-100:] if (extra/'alerts.jsonl').exists() else []:
   ea=json.loads(line)
   if now-ea.get('time',0)<600 and 'stall:' in ea.get('key',''):add('all184:'+ea['key'].split(':reminder:')[0],ea)
 stage7=WIKI/'artifacts/stage7-stage5-20260918'
 sp=read(stage7/'pipeline-progress.json',{});sh=read(stage7/'monitor-heartbeat.json',{})
 if sp.get('phase')=='failed':add('stage7-failed:'+str(sp.get('time')),sp)
 elif sp.get('phase') not in ('complete',None):
  if now-sh.get('time',0)>600:add('stage7-monitor-stale',sh)
  elif sh.get('service')!='active':add('stage7-inactive:'+str(sp.get('time')),sh)
  for line in (stage7/'alerts.jsonl').read_text().splitlines()[-100:] if (stage7/'alerts.jsonl').exists() else []:
   sa=json.loads(line)
   if now-sa.get('time',0)<600 and 'stall:' in sa.get('key',''):add('stage7:'+sa['key'].split(':reminder:')[0],sa)
 composition=WIKI/'artifacts/contextual-roi-comp-20260919'
 sp=read(composition/'pipeline-progress.json',{});sh=read(composition/'monitor-heartbeat.json',{})
 if sp.get('phase')=='failed':add('composition-failed:'+str(sp.get('time')),sp)
 elif sp.get('phase') not in ('complete',None):
  if now-sh.get('time',0)>600:add('composition-monitor-stale',sh)
  elif sh.get('service')!='active':add('composition-inactive:'+str(sp.get('time')),sh)
  for line in (composition/'alerts.jsonl').read_text().splitlines()[-100:] if (composition/'alerts.jsonl').exists() else []:
   sa=json.loads(line)
   if now-sa.get('time',0)<600 and 'stall:' in sa.get('key',''):add('composition:'+sa['key'].split(':reminder:')[0],sa)
 contextgate=WIKI/'artifacts/contextual-gate-20260919'
 sp=read(contextgate/'pipeline-progress.json',{});sh=read(contextgate/'monitor-heartbeat.json',{})
 if sp.get('phase')=='failed':add('contextgate-failed:'+str(sp.get('time')),sp)
 elif sp.get('phase') not in ('complete',None):
  if now-sh.get('time',0)>600:add('contextgate-monitor-stale',sh)
  elif sh.get('service')!='active':add('contextgate-inactive:'+str(sp.get('time')),sh)
  for line in (contextgate/'alerts.jsonl').read_text().splitlines()[-100:] if (contextgate/'alerts.jsonl').exists() else []:
   sa=json.loads(line)
   if now-sa.get('time',0)<600 and 'stall:' in sa.get('key',''):add('contextgate:'+sa['key'].split(':reminder:')[0],sa)
 langctl=WIKI/'artifacts/contextual-roi-langctl-20260919'
 sp=read(langctl/'pipeline-progress.json',{});sh=read(langctl/'monitor-heartbeat.json',{})
 if sp.get('phase')=='failed':add('langctl-failed:'+str(sp.get('time')),sp)
 elif sp.get('phase') not in ('complete',None):
  if now-sh.get('time',0)>600:add('langctl-monitor-stale',sh)
  elif sh.get('service')!='active':add('langctl-inactive:'+str(sp.get('time')),sh)
  for line in (langctl/'alerts.jsonl').read_text().splitlines()[-100:] if (langctl/'alerts.jsonl').exists() else []:
   sa=json.loads(line)
   if now-sa.get('time',0)<600 and 'stall:' in sa.get('key',''):add('langctl:'+sa['key'].split(':reminder:')[0],sa)
 dcb=WIKI/'artifacts/contextual-roi-dcb-20260921'
 sp=read(dcb/'pipeline-progress.json',{});sh=read(dcb/'monitor-heartbeat.json',{})
 if sp.get('phase')=='failed':add('dcb-failed:'+str(sp.get('time')),sp)
 elif sp.get('phase') not in ('complete',None):
  if now-sh.get('time',0)>600:add('dcb-monitor-stale',sh)
  elif sh.get('service')!='active':add('dcb-inactive:'+str(sp.get('time')),sh)
  for line in (dcb/'alerts.jsonl').read_text().splitlines()[-100:] if (dcb/'alerts.jsonl').exists() else []:
   sa=json.loads(line)
   if now-sa.get('time',0)<600 and 'stall:' in sa.get('key',''):add('dcb:'+sa['key'].split(':reminder:')[0],sa)
 concat=WIKI/'artifacts/contextual-roi-concat-20260924'
 sp=read(concat/'pipeline-progress.json',{});sh=read(concat/'monitor-heartbeat.json',{})
 if sp.get('phase')=='failed':add('concat-failed:'+str(sp.get('time')),sp)
 elif sp.get('phase') not in ('complete',None):
  if now-sh.get('time',0)>600:add('concat-monitor-stale',sh)
  elif sh.get('service')!='active':add('concat-inactive:'+str(sp.get('time')),sh)
  for line in (concat/'alerts.jsonl').read_text().splitlines()[-100:] if (concat/'alerts.jsonl').exists() else []:
   sa=json.loads(line)
   if now-sa.get('time',0)<600 and any(k in sa.get('key','') for k in ['stall:','log-error:','checkpoint:']):add('concat:'+sa['key'].split(':reminder:')[0],sa)
 r=read(WIKI/'artifacts/research-monitor/heartbeat.json',{})
 if now-r.get('checked_unix',0)>600:add('ncshare-monitor-stale',{'last':r.get('checked_utc')})
 else:
  live_jobs=r.get('jobs',[])
  for line in r.get('accounting','').splitlines():
   f=line.split('|')
   if (len(f)>1 and f[1].split()[0] in ['FAILED','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','TIMEOUT']
       and not resolved_accounting(line) and not continued_accounting(line,live_jobs)):
    add('cluster-job:'+f[0],line)
  for j in r.get('jobs',[]):
   if 'DependencyNeverSatisfied' in j.get('reason',''):add('dependency:'+j['job'],j)
  ap=WIKI/'artifacts/research-monitor/alerts.jsonl'
  for line in ap.read_text().splitlines()[-100:] if ap.exists() else []:
   a=json.loads(line)
   try:t=datetime.datetime.fromisoformat(a['time']).timestamp()
   except (KeyError,ValueError):continue
   if now-t<600 and a.get('severity') in ['error','warning']:add('remote-alert:'+a['id'],a)
 b=read(WIKI/'artifacts/stage56-full/checkpoint-backup/verified.json',{})
 if now-b.get('checked_unix',0)>14400:add('checkpoint-verification-stale',{'last_verified':b.get('checked_unix')})
 return list({i['key']:i for i in issues}.values())

def launch(folder,issues,smoke=False):
 prompt=(CODE/'repair-prompt.md').read_text()+'\nIncident snapshot:\n'+json.dumps(issues,indent=2)
 if smoke:
  prompt='Integration smoke test ONLY. Do not inspect or modify real studies or services. Write the text OK to '+str(folder/'smoke-ok.txt')+' using a tool. Then run the read-only command ssh -o BatchMode=yes -o ConnectTimeout=10 ncshare true using automatic approval review if needed. Report both results. Do not change permissions or bypass safeguards.'
 (folder/'prompt.txt').write_text(prompt)
 cmd=['/home/brandon/.local/bin/codex','exec','--approve-for-me','-C',str(WIKI),'--add-dir','/data/repos/ROAD_Reason/research/post-proposal','--json','-o',str(folder/'report.md'),'-']
 env=os.environ.copy()
 for k in ['CODEX_THREAD_ID','CODEX_SANDBOX','CODEX_INTERNAL_ORIGINATOR_OVERRIDE']:env.pop(k,None)
 with (folder/'events.jsonl').open('w') as out,(folder/'stderr.log').open('w') as err:
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,text=True,env=env,start_new_session=True)
  try:p.communicate(prompt,timeout=300 if smoke else 1800)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   return {'exit_code':p.returncode,'timed_out':True}
 return {'exit_code':p.returncode,'timed_out':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inspect',action='store_true');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'dispatch.lock').open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  now=time.time();issues=collect(now);state=read(ROOT/'state.json',{})
  atomic(ROOT/'heartbeat.json',{'time':now,'issues':issues,'mode':'inspect' if a.inspect else 'monitor'})
  if a.inspect:print(json.dumps(issues,indent=2));return
  eligible=[i for i in issues if now-state.get(i['key'],{}).get('last_attempt',0)>3600 and state.get(i['key'],{}).get('attempts',0)<3]
  exhausted=[i for i in issues if state.get(i['key'],{}).get('attempts',0)>=3]
  if exhausted:
   atomic(ROOT/'needs-attention.json',{'time':now,'issues':exhausted,'reason':'Three automatic sessions attempted; manual review required before more retries.'})
   marker=ROOT/'last-exhausted-alert'
   if not marker.exists() or now-marker.stat().st_mtime>900:notify('Repair retry limit reached. See '+str(ROOT/'needs-attention.json'));marker.touch()
  else:(ROOT/'needs-attention.json').unlink(missing_ok=True)
  if not a.smoke and not eligible:return
  folder=ROOT/('smoke-' if a.smoke else 'incident-')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');folder.mkdir(parents=True)
  if not a.smoke:
   for i in eligible:
    old=state.get(i['key'],{});state[i['key']]={'last_attempt':now,'attempts':old.get('attempts',0)+1}
   atomic(ROOT/'state.json',state);notify('Investigating an experiment incident; report: '+str(folder/'report.md'))
  atomic(ROOT/'active.json',{'started':now,'folder':str(folder),'issues':eligible,'smoke':a.smoke})
  result=launch(folder,eligible,a.smoke);result.update(finished=time.time(),folder=str(folder),smoke=a.smoke);atomic(folder/'result.json',result);atomic(ROOT/'last-run.json',result);(ROOT/'active.json').unlink(missing_ok=True)
  if not a.smoke:notify('Repair session finished. Read verified outcome: '+str(folder/'report.md'))
  print(json.dumps(result))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  atomic(ROOT/'dispatcher-error.json',{'time':time.time(),'error':repr(exc)})
  notify('Repair dispatcher failed: '+str(exc)[:250]);raise
