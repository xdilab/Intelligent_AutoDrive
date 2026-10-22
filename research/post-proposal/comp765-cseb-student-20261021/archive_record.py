"""Copy compact verified study records into the source repository; no upload."""
from common import *
import shutil,tarfile

def main():
 verify();assert read(ROOT/'final-audit.json')['passed']
 dest=HERE/'results';dest.mkdir(exist_ok=True)
 for source in (ROOT/'results').iterdir():
  if source.is_file():shutil.copy2(source,dest/source.name)
 records=dest/'records';records.mkdir(exist_ok=True)
 names=['freeze.json','split-manifest.json','audit.json','provenance.json','data-manifest.json','data-audit.json','detector-history-audit.json','annotation-mapping-audit.json','official-assets-audit.json','preflight.json','evaluation-preflight.json','teacher-comparator.json','teacher-numerical-parity-audit.json','preparation-complete.json','training-complete.json','inference-complete.json','inference-benchmark.json','evaluation-launch-estimate.json','runtime-launch.json','study-complete.json','final-audit.json','benchmark.json']
 for name in names:
  source=ROOT/name;assert source.exists(),source;shutil.copy2(source,records/name)
 for seed in [0,1,2]:
  shutil.copy2(ROOT/f'order-seed{seed}.json',records/f'order-seed{seed}.json')
  for condition in ['supervised','distilled']:
   run=f'{condition}-seed{seed}';target=records/run;target.mkdir(exist_ok=True)
   for name in ['history.json','training-complete.json','detector-results.json']:shutil.copy2(ROOT/'runs'/run/name,target/name)
 shutil.copy2(ROOT/'bundle-seed0/bundle.json',records/'bundle.json')
 for name in ['checkpoints.json','alerts.json','heartbeat.json','self-test.json']:
  shutil.copy2(ROOT/'monitor'/name,records/('monitor-'+name))
 with tarfile.open(records/'evaluation-input-hashes.tar.gz','w:gz') as tar:
  for source in sorted((ROOT/'evaluation-inputs').glob('*.json')):tar.add(source,arcname=source.name)
 with tarfile.open(records/'continuous-runtime-reports.tar.gz','w:gz') as tar:
  for source in sorted((ROOT/'runtime').glob('continuous-*/report.json')):tar.add(source,arcname=source.parent.name+'/report.json')
 save(dest/'archive.json',dict(time=time.time(),source_root=str(ROOT),scope='Compact verified records only; weights, training caches and prediction arrays remain in the server run root. No remote publication.',files={str(x.relative_to(dest)):sha(x) for x in sorted(dest.rglob('*')) if x.is_file() and x.name!='archive.json'}))
 print('Archived verified record:',dest,flush=True)
if __name__=='__main__':main()
