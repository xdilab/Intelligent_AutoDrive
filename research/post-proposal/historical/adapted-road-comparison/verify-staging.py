"""Release preflight when every atomically transferred frame has its expected size."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path('/work/bbyrd1/adapted-road-20260911');manifest=json.loads((root/'code/frame-inventory.json').read_text());end=time.monotonic()+12*3600
while True:
 missing=0
 for rel,size in manifest['files']:
  p=root/'frames'/rel
  try:ok=p.stat().st_size==size
  except FileNotFoundError:ok=False
  missing+=not ok
 print(json.dumps({'frames_remaining':missing,'total':len(manifest['files'])}),flush=True)
 if not missing:break
 if time.monotonic()>end:raise TimeoutError('Raw-frame staging not complete; smoke remains held')
 time.sleep(60)
p=root/'road_waymo_trainval_v1.1.json';h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==manifest['annotation_sha256']
record={'frames':len(manifest['files']),'bytes':sum(n for _,n in manifest['files']),'frame_inventory_sha256':hashlib.sha256((root/'code/frame-inventory.json').read_bytes()).hexdigest(),'annotation_sha256':h.hexdigest(),'verification':'every path and byte count; rsync atomic transfer; annotation SHA-256','completed_unix':time.time()}
(root/'transfer-complete.json').write_text(json.dumps(record,indent=2)+'\n')
ids=json.loads((root/'results/jobs.json').read_text());subprocess.run(['scontrol','release',ids['smoke']],check=True);print('STAGING VERIFIED; SMOKE RELEASED',flush=True)
