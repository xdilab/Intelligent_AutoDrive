"""Wait for the bounded archive download, verify SHA-256, then extract regular files safely."""
import hashlib,json,tarfile,time
from pathlib import Path
root=Path('/work/bbyrd1/bddx-pilot-20260911')
p=root/'data/BDDX_Processed.tar.part';expected_size=6168084480;expected_sha='47271890c1a1631319cb3978694abd9a302af2a36632d83846c175b19d22335a'
deadline=time.monotonic()+1800
while not p.exists() or p.stat().st_size!=expected_size:
 if time.monotonic()>deadline:raise TimeoutError('BDD-X download did not finish within 30 minutes')
 time.sleep(10)
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==expected_sha,'Archive checksum mismatch'
with tarfile.open(p) as tar:
 for m in tar:
  if not m.isfile():continue
  target=(root/'data'/m.name).resolve()
  assert target.is_relative_to((root/'data').resolve()),m.name
  target.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(m) as src,target.open('wb') as dst:
   while b:=src.read(1024*1024):dst.write(b)
(root/'data-ready.json').write_text(json.dumps({'sha256':h.hexdigest(),'bytes':expected_size,'source':'https://huggingface.co/datasets/javyduck/SafeAuto-BDDX/resolve/main/BDDX_Processed.tar'})+'\n')
print('DATA READY',flush=True)
