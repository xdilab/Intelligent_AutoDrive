"""Official evaluator fixture checks and fresh-image runtime replay before AP."""
from common import *
from evaluate import metric,step,verify_teacher,evaluation_signature
from inference import StudentInference
from PIL import Image
import shutil
def export(seed=0):
 dest=ROOT/'bundle-seed0';run=ROOT/'runs'/f'distilled-seed{seed}';ckpath=run/'best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);done=read(run/'training-complete.json');assert sha(ckpath)==done['checkpoint_sha256']
 if dest.exists():assert read(dest/'bundle.json')['student_checkpoint_sha256']==sha(ckpath);return dest
 dest.mkdir();p.atomic_save(ck['model'],dest/'student.pt');shutil.copy2(p.WEIGHTS,dest/'backbone.pth');shutil.copy2(p.YOLO,dest/'yolo.pt');shutil.copy2(OLD/'model.py',dest/'model.py');shutil.copy2(HERE/'standalone.py',dest/'run.py')
 save(dest/'bundle.json',dict(format=1,architecture='shared-resnet50-layer3-scene-gru256',sampling_hz=10,labels=read(EXPANDED/'manifest.json')['labels'],model_sha256=sha(OLD/'model.py'),files_sha256={n:sha(dest/n) for n in ['student.pt','backbone.pth','yolo.pt','model.py','run.py']},student_checkpoint_sha256=sha(ckpath),teacher_checkpoint_sha256=sha(p.TEACHER),training_manifest_sha256=sha(ROOT/'data-manifest.json'),condition='distilled',epoch=ck['epoch'],study='COMP765 causal distillation20261021',seed=0,selection='Prespecified seed0, no finalAP-based selection'))
 return dest
def main():
 setup();verify();from data_audit import main as audit_data
 audit_data();assert read(ROOT/'data-audit.json')['complete'];assert read(ROOT/'annotation-mapping-audit.json')['passed'];checks={};official=read(INPUTS['frames']);verify_teacher(official);checks['teacher_comparator_exact_identity']=True
 # Shared official function: missing GT is FN, higher-ranked false detection is FP.
 gt=[np.array([[0,0,1,1,0]],np.float32)];good=np.array([[0,0,1,1,.9]],np.float32);bad=np.array([[2,2,3,3,.95],[0,0,1,1,.9]],np.float32)
 _,a,_=metric.core['evaluate_detections'](gt,[[good]],['event'],iou_thresh=.5)
 _,b,_=metric.core['evaluate_detections'](gt,[[bad]],['event'],iou_thresh=.5)
 assert np.isclose(a[0],100) and np.isclose(b[0],50);checks['official_AP_fixture']=[float(a[0]),float(b[0])]
 # Same chosen checkpoint, fresh detector, same current image, full state replay.
 bundle=export();runner=StudentInference(bundle,ROOT/'yolo-settings-replay');model=Student().cuda().eval();model.load_state_dict(torch.load(ROOT/'runs/distilled-seed0/best.pt',map_location='cpu',weights_only=False)['model']);states={'m':None};splits=read(ROOT/'split-manifest.json');video=splits['groups']['train'][0];start=splits['start_frames'][video];maxdiff=0.;rows=0
 from train import sequence
 cached_record=next(r for r in read(ROOT/'data-manifest.json')['sequences'] if r['video']==video)
 cached=[to_gpu(r) for r in torch.load(cached_record['path'],map_location='cpu',weights_only=True)]
 with torch.inference_mode():
  cached_logits,_=sequence(model,cached);cached_scores=p.detector_scores(cached_logits.sigmoid(),torch.cat([r['q'] for r in cached])).cpu().numpy()
 fresh_scores=[]
 for fid in range(start,start+32):
  image=np.array(Image.open(image_path(video,fid)).convert('RGB'));out=runner.predict_frame(image);fresh_scores.append(out['scores'])
  result=step({'m':model},states,runner.backbone,image,torch.from_numpy(out['boxes']).cuda(),torch.from_numpy(out['agents']).cuda(),torch.from_numpy(out['confidence']).cuda())['m']
  diff=float(np.max(np.abs(result-out['scores']))) if result.size else 0.;maxdiff=max(maxdiff,diff);rows+=len(result);assert diff<=1e-7
 cache_diff=float(np.max(np.abs(cached_scores-np.concatenate(fresh_scores))));assert cache_diff<=1e-7
 checks['training_cache_to_fresh_inference_max_abs']=cache_diff
 checks['checkpoint_bundle_evaluator_replay']=dict(video=video,frames=32,proposals=rows,max_abs=maxdiff)
 assert all(name in ['student.pt','backbone.pth','yolo.pt','model.py','run.py'] for name in read(bundle/'bundle.json')['files_sha256']);checks['no_teacher_in_bundle']=True
 from standalone import Runner
 portable=Runner(bundle,ROOT/'yolo-settings-portable');runner.reset()
 for fid in range(start,start+8):
  image=np.array(Image.open(image_path(video,fid)).convert('RGB'));a=runner.predict_frame(image);b=portable.predict_frame(image)
  assert all(np.array_equal(a[k],b[k]) for k in a)
 checks['standalone_runner_exact_replay']=True
 save(ROOT/'evaluation-preflight.json',dict(passed=True,time=time.time(),checks=checks,evaluation_signature=evaluation_signature(),bundle=str(bundle)));progress('evaluation-preflight-passed',checks=checks)
if __name__=='__main__':main()
