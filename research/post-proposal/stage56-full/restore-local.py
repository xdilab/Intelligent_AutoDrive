"""Restore the exact original frame inventory; run on Brandon's workstation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,time
out=Path('/data/repos/wiki/artifacts/stage56-full/recovery');inventory=json.loads((out/'frame-inventory.json').read_text());source=Path('/data/datasets/ROAD_plusplus/rgb-images');out.mkdir(parents=True,exist_ok=True)
for rel,size in inventory['files']:assert (source/rel).stat().st_size==size,rel
for i in range(8):(out/f'files-{i}.txt').write_text(''.join(rel+'\n' for rel,_ in inventory['files'][i::8]))
def restore(i):
 for attempt in range(4):
  with (out/f'transfer-{i}.log').open('a') as log:
   rc=subprocess.run(['rsync','-rt','--partial','--stats','--files-from='+str(out/f'files-{i}.txt'),str(source)+'/','ncshare:/work/bbyrd1/stage56-full-20260914/frames/'],stdout=log,stderr=subprocess.STDOUT).returncode
  if rc==0:return i
  time.sleep(10)
 raise RuntimeError(f'Frame shard {i} failed after retries')
with ThreadPoolExecutor(max_workers=8) as pool:
 for i in pool.map(restore,range(8)):print('completed frame shard',i,flush=True)
