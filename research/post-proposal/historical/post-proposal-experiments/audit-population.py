import pickle,json,hashlib,time
from pathlib import Path
import numpy as np
E=Path('/data/repos/ROAD_Reason/experiments');out=Path(__file__).parent
print('Loading detector records',flush=True)
y=pickle.load(open(E/'exp11_yolo/dets_v8x_best_val_fullcand.pkl','rb'))['records']
p=pickle.load(open(E/'exp11_yolo/dets_i3d_val_fullcand.pkl','rb')); b=p['records']; labels=p['labels']
def key(stem):
 v,f=stem.rsplit('_',1);return v+'/'+str(int(f))
shared=sorted(s for s in y if key(s) in b)
print('Loading crop cache',flush=True)
p=pickle.load(open(E/'exp12_phrase_head/crop_full/crop_feats_val.pkl','rb'));f=p['feats']
missing=[s for s in shared if s not in f];mismatch=[s for s in shared if s in f and len(y[s]['boxes_xyxyn'])!=len(f[s])]
report={'yolo_frames':len(y),'i3d_frames':len(b),'shared_frames':len(shared),'crop_frames':len(f),'crop_meta':p.get('meta'),'missing':[{'stem':s,'yolo_boxes':len(y[s]['boxes_xyxyn']),'gt_boxes':len(b[key(s)]['gt']['boxes']) if b[key(s)].get('gt') is not None else 0} for s in missing],'mismatches':mismatch,'nonfinite_frames':[]}
for s in shared:
 if s in f and not np.isfinite(f[s]).all():report['nonfinite_frames'].append(s)
manifest={'frames':shared,'frame_sha256':hashlib.sha256(('\n'.join(shared)+'\n').encode()).hexdigest(),'candidate_sha256':hashlib.sha256(b''.join(s.encode()+np.asarray(y[s]['boxes_xyxyn']).tobytes()+np.asarray(y[s]['conf']).tobytes()+np.asarray(y[s]['cls']).tobytes() for s in shared)).hexdigest(),'labels':labels}
(out/'population-audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n');(out/'shared-frames.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(report,default=str),flush=True)
