"""Stable atomic-checkpoint snapshots on NCShare, downloaded and validated locally."""
from pathlib import Path
import subprocess,json,time,hashlib,shutil
import torch
OUT=Path('/data/repos/wiki/artifacts/stage56-full/checkpoint-backup');OUT.mkdir(parents=True,exist_ok=True)
REMOTE=r'''
from pathlib import Path
import os,json,time
r=Path('/work/bbyrd1/stage56-full-20260914');o=r/'checkpoint-backup';o.mkdir(exist_ok=True)
records=[]
for run in sorted((r/'runs').iterdir()):
 if not run.is_dir():continue
 for p in run.glob('*.pt'):
  if p.name not in ['resume.pt','best.pt'] and not p.name.startswith('epoch-'):continue
  target=o/run.name/p.name;target.parent.mkdir(exist_ok=True);tmp=target.with_suffix('.link')
  if tmp.exists():tmp.unlink()
  os.link(p,tmp);tmp.replace(target)
  records.append({'path':str(target.relative_to(o)),'bytes':target.stat().st_size,'mtime':target.stat().st_mtime})
(o/'snapshot.json').write_text(json.dumps({'time':time.time(),'files':records},indent=2))
print(len(records))
'''
def main():
 subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','ncshare','python3','-c',"'"+REMOTE.replace("'","'\\''")+"'"],check=True,timeout=90)
 # The remote .link entries are duplicate hard links used while publishing an
 # atomic snapshot.  Copy only canonical checkpoints so rsync does not expand
 # those links into several gigabytes of duplicate local files.
 subprocess.run(['rsync','-rt','--delay-updates','--timeout=120','--include=*/','--include=*.pt','--include=snapshot.json','--exclude=*','-e','ssh -o BatchMode=yes -o ConnectTimeout=15','ncshare:/work/bbyrd1/stage56-full-20260914/checkpoint-backup/',str(OUT)+'/'],check=True,timeout=1200)
 checks=[]
 for p in sorted(OUT.glob('*/resume.pt')):
  ck=torch.load(p,map_location='cpu',weights_only=False)
  assert all(k in ck for k in ['state','optimizer','epoch','position','steps','report','protocol','data_sha256'])
  assert ck['optimizer']['state'] and ck['state'] and ck['steps']>0
  assert all(torch.isfinite(v).all() for v in ck['state'].values())
  checks.append({'run':p.parent.name,'epoch_index':ck['epoch'],'position':ck['position'],'steps':ck['steps'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 assert len(checks)==6
 status={'checked_unix':time.time(),'passed':True,'runs':checks};(OUT/'verified.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True);(OUT/'backup-error.json').unlink(missing_ok=True)
 # Inference caches are append-only atomic .npy/.npz files. Exclude unfinished .tmp files.
 cache_errors=[]; cache_skipped=None
 # Inference caches are reproducible and optional; checkpoints are not.  Keep
 # the checkpoint verifier healthy instead of filling the filesystem while
 # large detector passes are active.
 if shutil.disk_usage(OUT).free < 10 * 1024**3:
  cache_skipped='low-space'
 for folder in [] if cache_skipped else ['predictions-epoch1','predictions','runs']:
  exists=subprocess.run(['ssh','-o','BatchMode=yes','ncshare',f'test -d /work/bbyrd1/stage56-full-20260914/{folder}'],timeout=30)
  if exists.returncode==1:continue
  exists.check_returncode()
  target=OUT/'inference'/folder;target.mkdir(parents=True,exist_ok=True)
  try:subprocess.run(['rsync','-rt','--delay-updates','--timeout=120','--include=*/','--include=*.npy','--include=*.npz','--include=provenance.json','--exclude=*','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',f'ncshare:/work/bbyrd1/stage56-full-20260914/{folder}/',str(target)+'/'],check=True,timeout=1200)
  except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:cache_errors.append({"folder":folder,"error":repr(exc)})
 (OUT/"inference-status.json").write_text(json.dumps({"time":time.time(),"passed":not cache_errors,"errors":cache_errors,"skipped":cache_skipped},indent=2))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  (OUT/'backup-error.json').write_text(json.dumps({'time':time.time(),'error':repr(exc)}))
  subprocess.run(['notify-send','Research checkpoint backup failed',str(exc)[:300]])
  raise
