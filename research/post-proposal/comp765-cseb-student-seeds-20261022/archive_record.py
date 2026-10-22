"""Archive compact verified evidence locally, without weights or uploads."""
from common import *
import shutil,tarfile
def main():
 assert read(ROOT/'final-audit.json')['passed'];dest=HERE/'results';dest.mkdir(exist_ok=True);records=dest/'records';records.mkdir(exist_ok=True)
 for source in (ROOT/'results').iterdir():
  if source.is_file():shutil.copy2(source,dest/source.name)
 names=['freeze.json','original-record.json','split-manifest.json','data-manifest.json','preparation-complete.json','data-audit.json','official-assets-audit.json','annotation-mapping-audit.json','detector-history-audit.json','provenance.json','launch-estimate.json','benchmark.json','preflight.json','evaluation-preflight.json','input-parity.json','training-complete.json','inference-complete.json','study-complete.json','final-audit.json']
 for name in names:shutil.copy2(ROOT/name,records/name)
 for name in ['prediction-buffer-provenance.json','scratch-inference-receipt.json']:
  if (ROOT/name).exists():shutil.copy2(ROOT/name,records/name)
 for seed in range(10):
  source=ORIGINAL_ROOT if seed<3 else ROOT;shutil.copy2(source/f'order-seed{seed}.json',records/f'order-seed{seed}.json')
  for condition in ['supervised','distilled']:
   run=f'{condition}-seed{seed}';target=records/run;target.mkdir(exist_ok=True)
   for name in ['training-complete.json','history.json','detector-results.json']:shutil.copy2(source/'runs'/run/name,target/name)
 for name in ['heartbeat.json','alerts.json','checkpoints.json','self-test.json']:shutil.copy2(ROOT/'monitor'/name,records/('monitor-'+name))
 for name,source in [('official-frames.json',INPUTS['frames']),('train-counts.json',INPUTS['counts']),('environment-freeze.txt',ORIGINAL_ROOT/'environment-freeze.txt')]:shutil.copy2(source,records/name)
 with tarfile.open(records/'evaluation-input-hashes.tar.gz','w:gz') as tar:
  for source in sorted((ROOT/'evaluation-inputs').glob('*.json')):tar.add(source,arcname=source.name)
 save(dest/'archive.json',dict(time=time.time(),run_root=str(ROOT),original_root=str(ORIGINAL_ROOT),scope='All20 per-run results and histories; no weights or visual caches copied; no remote publication.',files={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='archive.json'}));print('ARCHIVE COMPLETE',dest)
if __name__=='__main__':main()
