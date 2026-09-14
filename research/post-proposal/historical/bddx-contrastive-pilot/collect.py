"""Collect small pilot reports locally for up to 72 hours; checkpoints stay on NCShare."""
import subprocess,time
from pathlib import Path
root='/work/bbyrd1/bddx-pilot-20260911';out=Path(__file__).resolve().parent/'collected';out.mkdir(exist_ok=True)
def run(args):return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
end=time.monotonic()+72*3600
while time.monotonic()<end:
 try:
  state=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare','sacct -j 729086,729087 -X --format=JobID,State,Elapsed,ExitCode -n'])
  (out/'status.txt').write_text(state.stdout)
  for name in ['launch.json','data-ready.json','manifest.json','run-seed0/report.json','smoke-data/report.json']:
   r=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare',f'test -f {root}/{name} && cat {root}/{name}'])
   if r.returncode==0:
    p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(r.stdout)
  run(['rsync','-rt','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',f'ncshare:{root}/logs/',str(out/'logs')+'/'])
  print(time.strftime('%Y-%m-%d %H:%M:%S'),state.stdout,flush=True)
  if '729087' in state.stdout and any('729087' in line and ('COMPLETED' in line or 'FAILED' in line or 'CANCELLED' in line or 'TIMEOUT' in line or 'OUT_OF_ME' in line) for line in state.stdout.splitlines()):break
  if any('729086' in line and ('FAILED' in line or 'TIMEOUT' in line or 'OUT_OF_ME' in line) for line in state.stdout.splitlines()):break
 except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
 time.sleep(120)
