from pathlib import Path
import json,pickle,numpy as np,time
r=Path('/work/bbyrd1/stage56-full-20260914');src=Path('/work/bbyrd1/proposal-study-20260910');deploy=json.loads((r/'results/eval-input-optimization-deployment.json').read_text());d=pickle.load((src/'data/dets_v8x_best_val_fullcand.pkl').open('rb'))['records'];out={}
for condition in ['classification','contrastive']:
 p=r/'predictions-epoch1'/('stage5-'+condition);files=sorted(p.glob('*.npy'));split=deploy['preserved'][condition]['frames']
 def measure(ff):
  stamps=[f.stat().st_mtime for f in ff]
  if len(ff)<2:return {'frames':len(ff)}
  seconds=stamps[-1]-stamps[0];crops=sum(len(d[f.stem]['boxes_xyxyn']) for f in ff[1:])
  return {'frames':len(ff)-1,'crops':crops,'seconds':seconds,'crops_per_second':crops/seconds,'frames_per_second':(len(ff)-1)/seconds,'first':ff[0].stem,'last':ff[-1].stem}
 out[condition]={'before':measure(files[max(0,split-1001):max(0,split-100)]),'after':measure(files[split+2:]),'total_saved_frames':len(files)}
out['time']=time.time();(r/'results/eval-input-throughput.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
