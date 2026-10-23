"""Remove only owned regenerable buffers after results, backups and bundle audit."""
from common import *
import shutil
from evaluate import SCRATCH

def main():
 assert read(ROOT/'final-audit.json')['passed'];assert not (ROOT/'temporary-cleanup.json').exists();records=[]
 for seed in [0,1,2]:
  for c in CONDITIONS:
   assert (ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json').exists();f=SCRATCH/f'{c}-seed{seed}.npy';assert f.is_file() and not f.is_symlink();a=np.load(f,mmap_mode='r');assert a.shape==(6163968,184) and a.dtype==np.float32;del a
   records.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f),kind='official_scores'))
 receipt=SCRATCH/'inference.json';shutil.copy2(receipt,ROOT/'scratch-inference-receipt.json')
 for r in read(ROOT/'data-manifest.json')['sequences']:
  f=Path(r['prefix']['path']);assert f.parent==PREFIX and f.is_file() and not f.is_symlink();h=sha(f);assert h==r['prefix']['sha256'];records.append(dict(path=str(f),bytes=f.stat().st_size,sha256=h,kind='frozen_prefix'))
 save(ROOT/'temporary-cleanup.json',dict(time=time.time(),status='verified-before-removal',records=records,total_bytes=sum(r['bytes'] for r in records),regenerate='python prepare.py; scores regenerate from selected checkpoints with evaluate.py in a new scratch root; historical inputs/checkpoints preserved'))
 for r in records:Path(r['path']).unlink()
 receipt.unlink();SCRATCH.rmdir();PREFIX.rmdir();d=read(ROOT/'temporary-cleanup.json');d.update(status='removed-owned-regenerable-buffers',completed=time.time());save(ROOT/'temporary-cleanup.json',d);print('Removed',d['total_bytes']/2**30,'GiB of verified owned temporary buffers')
if __name__=='__main__':main()
