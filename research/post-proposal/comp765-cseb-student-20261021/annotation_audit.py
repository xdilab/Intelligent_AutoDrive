"""Independent name-remapping check of stored official GT; no AP computation."""
from common import *
import pickle
def main():
 torch.set_num_threads(2);ann=read(p.ROOT/'road_waymo_trainval_v1.1.json');official=read(INPUTS['frames']);payload=pickle.load(INPUTS['gt'].open('rb'));labels=official['labels'];lookups={h:{name:i for i,name in enumerate(labels[h])} for h in HEADS[1:]};scale=np.array([840,600,840,600],np.float32);selected=[]
 for video in read(ROOT/'split-manifest.json')['groups']['final']:
  keys=[k for k in official['frames'] if k.startswith(video+'_')];selected.extend([keys[0],keys[len(keys)//2],keys[-1]])
 count=0;max_error=0.;mismatches=[]
 for key in selected:
  video,fid=key.rsplit('_',1);frame=ann['db'][video]['frames'].get(str(int(fid)),{});stored=payload['records'][video+'/'+str(int(fid))]['gt'];items=[x for x in frame.get('annos',{}).values() if isinstance(x,dict) and x.get('box') is not None and len(x['box'])==4] if frame.get('annotated')==1 else []
  if not items:assert stored is None or len(stored['boxes'])==0;continue
  assert stored is not None and len(stored['boxes'])==len(items),key
  boxes=np.array([x['box'] for x in items],np.float32)*scale;error=float(np.max(np.abs(boxes-stored['boxes'])));max_error=max(max_error,error);assert error<1e-4,(key,error)
  for h in HEADS[1:]:
   expected=np.zeros((len(items),len(labels[h])),np.float32)
   for i,item in enumerate(items):
    for source_id in item.get(h+'_ids',[]):
     name=ann['all_'+h+'_labels'][source_id]
     if name in lookups[h]:expected[i,lookups[h][name]]=1
   if not np.array_equal(expected,stored[h]):mismatches.append(dict(frame=key,group=h,entries=int(np.count_nonzero(expected!=stored[h]))))
  count+=len(items)
 save(ROOT/'annotation-mapping-audit.json',dict(passed=not mismatches,time=time.time(),frames=len(selected),boxes=count,max_pixel_box_error=max_error,label_mismatches=mismatches,method='Independent annotation IDs -> master label strings -> curated output rows; compare to stored official GT; three fixed positions/video, no AP',sources_sha256={str(x):sha(x) for x in [HERE/'annotation_audit.py',PARENT.parents[1]/'experiments/exp4_retinamoon/dataloader.py',PARENT.parents[1]/'experiments/exp1_road_r/dataset.py']}));assert not mismatches,mismatches[:10];print('ANNOTATION REMAP VERIFIED',len(selected),count,max_error,flush=True)
if __name__=='__main__':main()
