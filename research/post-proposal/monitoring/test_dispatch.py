import importlib.util,json,tempfile,time
from pathlib import Path
p=Path(__file__).parent/'repair_dispatch.py';s=importlib.util.spec_from_file_location('dispatch',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
with tempfile.TemporaryDirectory() as td:
 m.WIKI=Path(td);m.CTX=m.WIKI/'artifacts/contextual-roi-20260917';now=time.time()
 def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
 put(m.CTX/'pipeline-progress.json',{'phase':'context-extraction','time':now})
 put(m.CTX/'monitor-heartbeat.json',{'time':now,'service':'active'})
 put(m.WIKI/'artifacts/research-monitor/heartbeat.json',{'checked_unix':now,'jobs':[],'accounting':''})
 put(m.WIKI/'artifacts/stage56-full/checkpoint-backup/verified.json',{'checked_unix':now})
 assert m.collect(now)==[]
 put(m.CTX/'pipeline-progress.json',{'phase':'failed','time':now,'error':'fixture only'})
 a=m.collect(now);assert len(a)==1 and a[0]['key'].startswith('context-failed:')
 put(m.CTX/'pipeline-progress.json',{'phase':'context-extraction','time':now})
 put(m.CTX/'monitor-heartbeat.json',{'time':now-700,'service':'active'})
 assert any(x['key']=='local-monitor-stale' for x in m.collect(now))
 put(m.CTX/'monitor-heartbeat.json',{'time':now,'service':'active'})
 put(m.WIKI/'artifacts/research-monitor/heartbeat.json',{'checked_unix':now,'accounting':'123|FAILED|1:0','jobs':[{'job':'124','reason':'DependencyNeverSatisfied'}]})
 assert {x['key'] for x in m.collect(now)}=={'cluster-job:123','dependency:124'}
print('PASS: healthy study does not launch, local failure/stale monitor and remote failure/dependency detected')
