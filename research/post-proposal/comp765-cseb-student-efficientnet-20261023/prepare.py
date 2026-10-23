"""Replace only frozen visual features, retaining all matched supervision."""
from common import *
from PIL import Image
import argparse
@torch.inference_mode()
def encode_sequence(rec,backbone):
 assert sha(rec['path'])==rec['sha256'];prior=torch.load(rec['path'],map_location='cpu',weights_only=True);seq=[];digests={}
 for row in prior:
  path=image_path(rec['video'],row['fid']);h=sha(path);assert h==rec['frame_sha256'][str(row['fid'])];digests[str(row['fid'])]=h
  image=np.array(Image.open(path).convert('RGB'));roi,scene=backbone(image,row['boxes']);new=dict(row,roi=roi.half().cpu(),scene=scene.half().cpu());assert torch.isfinite(new['roi']).all() and torch.isfinite(new['scene']).all();assert new['roi'].shape==(len(row['boxes']),1280) and new['scene'].shape==(1280,)
  for k in row:
   if k in ['roi','scene']:continue
   assert torch.equal(new[k],row[k]) if torch.is_tensor(row[k]) else new[k]==row[k]
  assert ('teacher' in new)==(rec['split']=='train');seq.append(new)
 return seq,digests

def main():
 setup();verify();src=read(ORIGINAL_ROOT/'data-manifest.json');records=read(ROOT/'data-manifest.json')['sequences'] if (ROOT/'data-manifest.json').exists() else [];byvideo={r['video']:r for r in records};backbone=SharedBackbone(WEIGHTS).cuda();started=time.perf_counter();(ROOT/'cache').mkdir(exist_ok=True)
 for rec in src['sequences']:
  if rec['video'] in byvideo:assert sha(byvideo[rec['video']]['path'])==byvideo[rec['video']]['sha256'];continue
  seq,digests=encode_sequence(rec,backbone);path=ROOT/'cache'/f"{rec['video']}.pt";p.atomic_save(seq,path)
  new=dict(rec,path=str(path),sha256=sha(path),source_cache=rec['path'],source_cache_sha256=rec['sha256'],current_image_sha256=digests,backbone_sha256=sha(WEIGHTS),unchanged_supervision_verified=True);records.append(new);byvideo[rec['video']]=new
  save(ROOT/'data-manifest.json',dict(signature=signature(),labels=src['labels'],groups=src['groups'],sequences=records));progress('preparation',completed_sequences=len(records),total=76,frames=len(records)*32,elapsed=time.perf_counter()-started)
 assert len(records)==76
 save(ROOT/'preparation-complete.json',dict(signature=signature(),data_manifest_sha256=sha(ROOT/'data-manifest.json'),original_data_manifest_sha256=sha(ORIGINAL_ROOT/'data-manifest.json'),time=time.time(),seconds=time.perf_counter()-started,frames=2432,teacher_targets_only_train=True,unchanged_supervision_verified=True))
 progress('prepared',frames=2432,seconds=time.perf_counter()-started)
if __name__=='__main__':main()
