"""Supervision identity, official metric fixtures, packaged checkpoint replay."""
from common import *
from evaluate import metric,step,verify_teacher,evaluation_signature
from standalone import Runner
from train import sequence
from PIL import Image
import shutil

def export():
 dest=ROOT/'bundle-seed0';ckpath=ROOT/'runs/distilled-seed0/best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert sha(ckpath)==read(ROOT/'runs/distilled-seed0/training-complete.json')['checkpoint_sha256']
 if dest.exists():assert read(dest/'bundle.json')['student_checkpoint_sha256']==sha(ckpath);return dest
 dest.mkdir();p.atomic_save(ck['model'],dest/'student.pt');shutil.copy2(WEIGHTS,dest/'backbone.pth');shutil.copy2(p.YOLO,dest/'yolo.pt');shutil.copy2(HERE/'model.py',dest/'model.py');shutil.copy2(HERE/'standalone.py',dest/'run.py')
 save(dest/'bundle.json',dict(format=1,architecture='shared-efficientnet-b0-scene-gru256',sampling_hz=10,labels=read(ROOT/'data-manifest.json')['labels'],files_sha256={n:sha(dest/n) for n in ['student.pt','backbone.pth','yolo.pt','model.py','run.py']},student_checkpoint_sha256=sha(ckpath),training_manifest_sha256=sha(ROOT/'data-manifest.json'),condition='distilled',epoch=ck['epoch'],seed=0,selection='Prespecified seed0, no final-AP selection',study='COMP765 EfficientNet follow-up20261023'))
 return dest

def main():
 setup();verify();checks={};official=read(INPUTS['frames']);verify_teacher(official)
 for rec in read(ROOT/'data-manifest.json')['sequences']:
  assert sha(rec['path'])==rec['sha256'] and sha(rec['source_cache'])==rec['source_cache_sha256'];a=torch.load(rec['path'],weights_only=True);b=torch.load(rec['source_cache'],weights_only=True)
  for x,y in zip(a,b):
   assert len(a)==len(b)==32 and set(x)==set(y)
   for k in x:
    if k not in ['roi','scene']:assert torch.equal(x[k],y[k]) if torch.is_tensor(x[k]) else x[k]==y[k]
 checks['all76_supervision_and_candidates_identical']=True
 gt=[np.array([[0,0,1,1,0]],np.float32)];good=np.array([[0,0,1,1,.9]],np.float32);bad=np.array([[2,2,3,3,.95],[0,0,1,1,.9]],np.float32)
 _,a,_=metric.core['evaluate_detections'](gt,[[good]],['event'],iou_thresh=.5);_,b,_=metric.core['evaluate_detections'](gt,[[bad]],['event'],iou_thresh=.5);assert np.isclose(a[0],100) and np.isclose(b[0],50);checks['official_AP_fixture']=[float(a[0]),float(b[0])]
 dest=export();runner=Runner(dest,ROOT/'yolo-settings-replay');model=Student().cuda().eval();model.load_state_dict(torch.load(ROOT/'runs/distilled-seed0/best.pt',map_location='cpu',weights_only=False)['model']);rec=read(ROOT/'data-manifest.json')['sequences'][0];raw=torch.load(rec['path'],weights_only=True);states={'m':None};fresh=[]
 with torch.inference_mode():
  seq=[to_gpu(v) for v in raw];z,_=sequence(model,seq);cached=p.detector_scores(z.sigmoid(),torch.cat([r['q'] for r in seq])).cpu().numpy()
 for r in raw:
  image=np.array(Image.open(image_path(rec['video'],r['fid'])).convert('RGB'));fresh.append(step({'m':model},states,runner.backbone,image,r['boxes'].cuda(),r['agents'].cuda(),r['q'].cuda())['m'])
 diff=float(np.max(np.abs(cached-np.concatenate(fresh))));assert diff<=1e-7;checks['cached_to_fresh_max_abs']=diff;states={'m':None};runner.reset();maxdiff=0.
 for r in raw[:8]:
  image=np.array(Image.open(image_path(rec['video'],r['fid'])).convert('RGB'));out=runner.predict_frame(image);z=step({'m':model},states,runner.backbone,image,torch.from_numpy(out['boxes']).cuda(),torch.from_numpy(out['agents']).cuda(),torch.from_numpy(out['confidence']).cuda())['m'];diff=float(np.max(np.abs(z-out['scores']))) if z.size else 0.;maxdiff=max(diff,maxdiff)
 assert maxdiff<=1e-7;checks['packaged_evaluator_max_abs']=maxdiff;checks['no_teacher_asset_in_bundle']=set(read(dest/'bundle.json')['files_sha256'])=={'student.pt','backbone.pth','yolo.pt','model.py','run.py'};assert checks['no_teacher_asset_in_bundle']
 save(ROOT/'evaluation-preflight.json',dict(passed=True,time=time.time(),checks=checks,evaluation_signature=evaluation_signature()));progress('evaluation-preflight-passed',checks=checks)
if __name__=='__main__':main()
