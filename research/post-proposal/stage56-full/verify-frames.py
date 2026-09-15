from pathlib import Path
import json,time,hashlib,datetime
root=Path('/work/bbyrd1/stage56-full-20260914');frames=root/'frames';inventory=json.loads((root/'code/frame-inventory.json').read_text());deadline=time.monotonic()+6*3600
while True:
 missing=0
 for rel,size in inventory['files']:
  p=frames/rel
  try:ok=p.stat().st_size==size
  except FileNotFoundError:ok=False
  missing+=not ok
 print('FILES_REMAINING',missing,'of',len(inventory['files']),flush=True)
 if not missing:break
 if time.monotonic()>deadline:raise TimeoutError('Frame restore incomplete')
 time.sleep(45)
expected={}
for split in ['train','dev']:
 with (root/f'data/{split}.jsonl').open() as f:
  for line in f:
   row=json.loads(line)
   for fid,digest in zip(row['fids'],row['frame_sha256']):
    rel=f'{row["video"]}/{fid:05d}.jpg'
    if rel in expected:assert expected[rel]==digest
    expected[rel]=digest
for i,(rel,digest) in enumerate(expected.items()):
 assert hashlib.sha256((frames/rel).read_bytes()).hexdigest()==digest,rel
 if i%10000==0:print('HASHED',i,'of',len(expected),flush=True)
report={'passed':True,'frame_root':str(frames),'inventory_files':len(inventory['files']),'bytes':sum(n for _,n in inventory['files']),'training_development_frame_hashes_verified':len(expected),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(root/'results/frame-recovery.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
