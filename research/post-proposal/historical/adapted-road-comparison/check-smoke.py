import json,os,pickle
from pathlib import Path
import numpy as np
root=Path(os.environ['COMPARISON_ROOT']);d=pickle.load((root/'smoke/crop_feats_val.shard0of4.pkl').open('rb'));old=pickle.load(Path('/work/bbyrd1/proposal-study-20260910/data/crop_feats_val.pkl').open('rb'))['feats'];report={}
for k,x in d['feats'].items():
 a=x.astype(float);b=old[k].astype(float);assert a.shape==b.shape
 if not len(a):continue
 cosine=np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)).clip(1e-12)
 assert cosine.min()>.999,'Original crop-feature parity failed'
 diff=float(np.mean(np.abs(a-d['adapted_feats'][k].astype(float))));assert diff>1e-6,'Adapted features unexpectedly unchanged'
 report[k]={'original_min_cosine':float(cosine.min()),'original_mae':float(np.mean(np.abs(a-b))),'adaptation_mae':diff}
assert report
(root/'results/smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
