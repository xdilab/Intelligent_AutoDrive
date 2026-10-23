"""Temporary exact-bfloat16 frozen-prefix cache, immutable labels and targets."""
from common import *
from training_model import Prefix
from PIL import Image
@torch.no_grad()
def encode_sequence(rec,backbone):
 assert sha(rec['path'])==rec['sha256'];rows=torch.load(rec['path'],map_location='cpu',weights_only=True);xs=[];digests={}
 for r in rows:
  path=image_path(rec['video'],r['fid']);h=sha(path);assert h==rec['frame_sha256'][str(r['fid'])];digests[str(r['fid'])]=h
  xs.append(backbone(np.array(Image.open(path).convert('RGB'))).cpu())
 x=torch.cat(xs);assert x.shape==(32,512,56,84) and x.dtype==torch.bfloat16 and torch.isfinite(x).all()
 return x,digests

def main():
 setup();verify();PREFIX.mkdir(exist_ok=True);records=[];b=Prefix().cuda();start=time.perf_counter()
 for rec in read(ORIGINAL_ROOT/'data-manifest.json')['sequences']:
  path=PREFIX/(rec['video']+'.pt');receipt=ROOT/'prefix-receipts'/(rec['video']+'.json')
  if path.exists() and receipt.exists():
   d=read(receipt);assert sha(path)==d['sha256'] and d['source_sha256']==rec['sha256']
  else:
   x,digests=encode_sequence(rec,b);p.atomic_save(x,path);d=dict(path=str(path),sha256=sha(path),source_sha256=rec['sha256'],current_image_sha256=digests);save(receipt,d)
  records.append(dict(rec,prefix=d));progress('prefix-preparation',sequences=len(records),total=76,seconds=time.perf_counter()-start)
 src=read(ORIGINAL_ROOT/'data-manifest.json');save(ROOT/'data-manifest.json',dict(signature=signature(),labels=src['labels'],groups=src['groups'],sequences=records))
 save(ROOT/'preparation-complete.json',dict(time=time.time(),seconds=time.perf_counter()-start,data_manifest_sha256=sha(ROOT/'data-manifest.json'),original_data_manifest_sha256=sha(ORIGINAL_ROOT/'data-manifest.json'),frames=2432,temporary_cache=str(PREFIX),regenerate_command='python prepare.py',teacher_targets_only_train=True))
if __name__=='__main__':main()
