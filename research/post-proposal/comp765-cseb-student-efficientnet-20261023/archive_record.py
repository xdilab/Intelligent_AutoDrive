"""Compact reproducible evidence, no model weights or image/cache duplication."""
from common import *
import shutil,tarfile

def main():
 assert read(ROOT/'final-audit.json')['passed'];dest=HERE/'results';dest.mkdir(exist_ok=True);rec=dest/'records';rec.mkdir(exist_ok=True)
 for f in (ROOT/'results').iterdir():
  if f.is_file():shutil.copy2(f,dest/f.name)
 for f in ROOT.glob('*.json'):shutil.copy2(f,rec/f.name)
 for sub in ['monitor','runtime']:
  for f in (ROOT/sub).rglob('*.json'):
   if 'yolo-settings' in str(f):continue
   target=rec/sub/f.relative_to(ROOT/sub);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
 for architecture,base in [('efficientnet',ROOT),('resnet',ORIGINAL_ROOT)]:
  for seed in [0,1,2]:
   for condition in ['supervised','distilled']:
    target=rec/f'{architecture}-{condition}-seed{seed}';target.mkdir(exist_ok=True)
    for name in ['detector-results.json','training-complete.json','history.json']:shutil.copy2(base/'runs'/f'{condition}-seed{seed}'/name,target/name)
 shutil.copy2(ROOT/'bundle-seed0/bundle.json',rec/'bundle.json')
 for name,key in [('official-frames.json','frames'),('train-counts.json','counts')]:shutil.copy2(INPUTS[key],rec/name)
 with tarfile.open(rec/'evaluation-input-hashes.tar.gz','w:gz') as tar:
  for f in sorted((ROOT/'evaluation-inputs').glob('*.json')):tar.add(f,arcname=f.name)
 save(dest/'archive.json',dict(time=time.time(),root=str(ROOT),files={str(f.relative_to(dest)):sha(f) for f in sorted(dest.rglob('*')) if f.is_file() and f.name!='archive.json'}));print('ARCHIVE COMPLETE',dest)
if __name__=='__main__':main()
