"""Selected checkpoint export and exact framewise inference parity."""
from common import *
from training_model import TrainModel,Prefix
from evaluate import metric,step,verify_teacher,evaluation_signature
from standalone import Runner
from PIL import Image
import shutil

def export(condition='tuned-distilled'):
 dest=ROOT/('bundle-seed0' if condition=='tuned-distilled' else 'bundle-frozen-seed0');ckpath=ROOT/'runs'/f'{condition}-seed0'/'best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert sha(ckpath)==read(ckpath.parent/'training-complete.json')['checkpoint_sha256']
 if dest.exists():assert read(dest/'bundle.json')['student_checkpoint_sha256']==sha(ckpath);return dest
 dest.mkdir();head={k[len('head.'):]:v for k,v in ck['model'].items() if k.startswith('head.')};backbone=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
 for k,v in ck['model'].items():
  if k.startswith('stage.'):backbone['layer3.'+k[len('stage.'):]]=v
 p.atomic_save(head,dest/'student.pt');p.atomic_save(backbone,dest/'backbone.pth');shutil.copy2(p.YOLO,dest/'yolo.pt');shutil.copy2(HERE/'model.py',dest/'model.py');shutil.copy2(HERE/'standalone.py',dest/'run.py')
 save(dest/'bundle.json',dict(format=1,architecture='shared-resnet50-layer3-scene-gru256',sampling_hz=10,labels=read(ROOT/'data-manifest.json')['labels'],files_sha256={n:sha(dest/n) for n in ['student.pt','backbone.pth','yolo.pt','model.py','run.py']},student_checkpoint_sha256=sha(ckpath),training_manifest_sha256=sha(ROOT/'data-manifest.json'),condition=condition,epoch=ck['epoch'],seed=0,selection='Prespecified seed0, no final-AP selection',study='COMP765 layer3 follow-up20261023'))
 return dest

def main():
 setup();verify();checks={};verify_teacher(read(INPUTS['frames']))
 for rec in read(ROOT/'data-manifest.json')['sequences']:assert sha(rec['path'])==rec['sha256']
 checks['original_supervision_caches_unchanged']=True
 gt=[np.array([[0,0,1,1,0]],np.float32)];good=np.array([[0,0,1,1,.9]],np.float32);bad=np.array([[2,2,3,3,.95],[0,0,1,1,.9]],np.float32)
 _,a,_=metric.core['evaluate_detections'](gt,[[good]],['event'],iou_thresh=.5);_,b,_=metric.core['evaluate_detections'](gt,[[bad]],['event'],iou_thresh=.5);assert np.isclose(a[0],100) and np.isclose(b[0],50);checks['AP_fixture']=[float(a[0]),float(b[0])]
 for condition in ['frozen-distilled','tuned-distilled']:
  dest=export(condition);runner=Runner(dest,ROOT/'yolo-settings-replay');model=TrainModel(condition.startswith('tuned')).cuda().eval();model.load_state_dict(torch.load(ROOT/'runs'/f'{condition}-seed0'/'best.pt',map_location='cpu',weights_only=False)['model']);prefix=Prefix().cuda();rec=read(ROOT/'data-manifest.json')['sequences'][0];rows=torch.load(rec['path'],weights_only=True);states={'m':None};runner.reset();diff=0.
  for r in rows[:8]:
   image=np.array(Image.open(image_path(rec['video'],r['fid'])).convert('RGB'));out=runner.predict_frame(image);z=step({'m':model},states,prefix,image,torch.from_numpy(out['boxes']).cuda(),torch.from_numpy(out['agents']).cuda(),torch.from_numpy(out['confidence']).cuda())['m'];diff=max(diff,float(np.max(np.abs(z-out['scores']))) if z.size else 0.)
  assert diff<=1e-7;checks[condition+'_packaged_evaluator_max_abs']=diff;assert set(read(dest/'bundle.json')['files_sha256'])=={'student.pt','backbone.pth','yolo.pt','model.py','run.py'}
  del runner,model,prefix;torch.cuda.empty_cache()
 save(ROOT/'evaluation-preflight.json',dict(passed=True,time=time.time(),checks=checks,evaluation_signature=evaluation_signature()));progress('evaluation-preflight-passed',checks=checks)
if __name__=='__main__':main()
