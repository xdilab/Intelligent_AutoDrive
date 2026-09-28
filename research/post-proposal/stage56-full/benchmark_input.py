import json,pickle,time,os
from pathlib import Path
import numpy as np
import torch
import base
from train import parts
from architecture import FullStage
from eval_input import ClipLoader,normalize_clip,crop_batch
root=Path('/work/bbyrd1/stage56-full-20260914');cfg=json.loads((root/'code/protocol.json').read_text());torch.set_num_threads(8)
src=Path(cfg['original_study']);keys=json.loads((src/'code/shared-frames.json').read_text())['frames'];d=pickle.load((src/'data/dets_v8x_best_val_fullcand.pkl').open('rb'))['records'];ann=json.loads(Path(cfg['annotation']).read_text())
chosen=[keys[5100],keys[5101],keys[5102]];loader=ClipLoader(cfg);records=[];sample=None
for s in chosen:
 v,f=s.rsplit('_',1);fid=int(f);nf=ann['db'][v]['numf'];fids=[min(max(fid-3+j,1),nf) for j in range(8)];b=d[s]['boxes_xyxyn'].tolist();row={'video':v,'fids':fids,'key_t':fids.index(fid),'boxes':b,'targets':np.zeros((len(b),184),np.float32),'frame_sha256':[base.sha(base.frame_root(cfg)/v/f'{x:05d}.jpg') for x in fids]}
 if not b:continue
 # Warm kernels and then compare every crop in a complete candidate frame.
 x,_=base.crops(next(parts(row,64)),cfg);del x;torch.cuda.synchronize()
 t=time.perf_counter();old=[]
 for part in parts(row,64):
  x,_=base.crops(part,cfg);old.append(x.cpu());del x
 torch.cuda.synchronize();old_seconds=time.perf_counter()-t
 t=time.perf_counter();ft=normalize_clip(loader.load(row));new=[]
 for j in range(0,len(b),64):
  x=crop_batch(ft,b[j:j+64]);new.append(x.cpu());del x
 del ft;torch.cuda.synchronize();new_seconds=time.perf_counter()-t
 error=max(float((a-b).abs().max()) for a,b in zip(old,new));assert error==0,error
 records.append({'key':s,'candidates':len(b),'old_preprocess_seconds':old_seconds,'new_preprocess_seconds':new_seconds,'max_crop_error':error})
 if sample is None:sample=(old[0][:8].clone(),new[0][:8].clone(),row['key_t'])
 del old,new
print('CROP_PARITY',json.dumps(records),flush=True)
m=FullStage(cfg,'classification','stage5');ck=torch.load(root/'runs/stage5-classification/epoch-1.pt',map_location='cpu',weights_only=False)
with torch.no_grad():
 for n,p in m.named_parameters():
  if n in ck['state']:p.copy_(ck['state'][n])
 with torch.autocast('cuda',dtype=torch.bfloat16):a=m(sample[0].cuda(),sample[2])[0];b=m(sample[1].cuda(),sample[2])[0]
 error=float((a-b).abs().max());assert error==0,error
report={'passed':True,'frames':records,'logit_max_error':error,'sample_candidates':len(sample[0]),'jpeg_reads':loader.reads,'jpeg_cache_hits':loader.hits,'time':time.time(),'scope':'Exact full-frame crop parity; same-checkpoint logit parity on 8 crops. Preprocessing timings include CPU copies and concurrent GPU workload; not end-to-end speedup.'}
(root/'results/eval-input-optimization-check.json').write_text(json.dumps(report,indent=2));print('PASS',json.dumps(report),flush=True)
