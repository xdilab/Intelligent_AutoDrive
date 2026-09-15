"""Local timer: pull persistent NCShare alerts and notify the user's desktop."""
from pathlib import Path
import subprocess,json,time,datetime
out=Path('/data/repos/wiki/artifacts/research-monitor');out.mkdir(parents=True,exist_ok=True)
def notify(title,message):
 return subprocess.run(['notify-send','--app-name=Research monitor',title,message],timeout=10).returncode==0
try:
 subprocess.run(['rsync','-rt','--timeout=30','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','ncshare:/work/bbyrd1/research-monitor/heartbeat.json','ncshare:/work/bbyrd1/research-monitor/alerts.jsonl',str(out)+'/'],check=True,capture_output=True,timeout=60)
except Exception as e:
 p=out/'connection-warning'
 if not p.exists() or time.time()-p.stat().st_mtime>3600:
  notify('Research monitor: connection problem','Cannot retrieve NCShare status. Monitoring visibility needs attention.');p.write_text(str(e))
 raise
p=out/'delivered.json';seen=set(json.loads(p.read_text())) if p.exists() else set()
for line in (out/'alerts.jsonl').read_text().splitlines():
 alert=json.loads(line)
 if alert['id'] not in seen:
  if notify('NCShare research: '+alert['severity'],alert['message']):seen.add(alert['id'])
p.write_text(json.dumps(sorted(seen)))
h=json.loads((out/'heartbeat.json').read_text())
if time.time()-h['checked_unix']>600:
 marker=out/'stale-warning'
 if not marker.exists() or time.time()-marker.stat().st_mtime>3600:
  notify('Research monitor: watchdog stale','NCShare watchdog has not checked in for over 10 minutes.');marker.touch()
(out/'last-pull.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
