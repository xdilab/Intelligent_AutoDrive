"""Parity, all new seed initializations, and representative measured work."""
from common import *
import ast,copy,gc
from PIL import Image
import train,evaluate,preflight

def function_ast(path,name):
 return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),include_attributes=False)

def main():
 setup();verify();checks={}
 for name in ['sequence','update','selection','capture','restore','load_data']:
  assert function_ast(HERE/'train.py',name)==function_ast(ORIGINAL_CODE/'train.py',name)
 for name in ['load_official','step','one_class']:
  assert function_ast(HERE/'evaluate.py',name)==function_ast(ORIGINAL_CODE/'evaluate.py',name)
 checks['training_and_score_functions_identical_AST']=True
 preflight.main()
 for seed in read(HERE/'protocol.json')['seeds']:
  setup(seed);a=Student().cuda();first=tensor_hash(a.state_dict());del a
  setup(seed);b=Student().cuda();assert first==tensor_hash(b.state_dict());del b
 checks['all_new_seeds_paired_initialization']=True
 data=train.load_data();measure=[]
 for condition,kd in [('supervised',0.),('distilled',1.)]:
  setup(999);model=Student().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
  train.update(model,opt,data['train'][0],kd);torch.cuda.synchronize();start=time.perf_counter()
  for seq in data['train']:train.update(model,opt,seq,kd)
  torch.cuda.synchronize();measure.append(dict(condition=condition,sequences=64,seconds=time.perf_counter()-start))
  del model,opt
 cached=data['train'][0]
 # Original selected model replay, using the exact same new evaluator's step.
 model=Student().cuda().eval();model.load_state_dict(torch.load(ORIGINAL_ROOT/'runs/distilled-seed0/best.pt',map_location='cpu',weights_only=False)['model']);backbone=SharedBackbone(p.WEIGHTS).cuda()
 with torch.inference_mode():
  logits,_=train.sequence(model,cached);expected=p.detector_scores(logits.sigmoid(),torch.cat([r['q'] for r in cached])).cpu().numpy()
  states={'oracle':None};actual=[];split=read(ROOT/'split-manifest.json');video=split['groups']['train'][0];start=split['start_frames'][video]
  for row,fid in zip(cached,range(start,start+32)):
   image=np.array(Image.open(image_path(video,fid)).convert('RGB'))
   actual.append(evaluate.step({'oracle':model},states,backbone,image,row['boxes'],row['agents'],row['q'])['oracle'])
 diff=float(np.max(np.abs(expected-np.concatenate(actual))));assert diff<=1e-7;checks['cached_to_fresh_original_checkpoint_max_abs']=diff
 del cached,data,model,backbone;gc.collect();torch.cuda.empty_cache()
 official,yr,offset,index=evaluate.load_official();evaluate.verify_teacher(official)
 models={str(i):Student().cuda().eval() for i in range(14)};backbone=SharedBackbone(p.WEIGHTS).cuda();frames=0;torch.cuda.synchronize();begin=time.perf_counter()
 for video in split['groups']['final'][:3]:
  states={k:None for k in models}
  for fid in range(1,split['final_last_frame'][video]+1):
   image=np.array(Image.open(image_path(video,fid)).convert('RGB'));ix=index.get((video,fid))
   if ix is None:box=torch.empty((0,4),device='cuda');agent=torch.empty(0,dtype=torch.long,device='cuda');q=torch.empty(0,device='cuda')
   else:
    rec=yr[official['frames'][ix]];box=torch.from_numpy(rec['boxes_xyxyn']).cuda().float();agent=torch.from_numpy(rec['cls']).cuda().long();q=torch.from_numpy(rec['conf']).cuda().float()
   evaluate.step(models,states,backbone,image,box,agent,q);frames+=1
 torch.cuda.synchronize();elapsed=time.perf_counter()-begin;gpu_peak=torch.cuda.max_memory_allocated()/1024**3
 del models,backbone;torch.cuda.empty_cache()
 # Full detector-only oracle checks unchanged q, geometry, ordering and AP.
 evaluate.metric.SCORES=[np.broadcast_to(yr[k]['conf'].astype(np.float32)[:,None],(len(yr[k]['conf']),184)) for k in official['frames']]
 begin=time.perf_counter();g,c,ap,detail=evaluate.metric.one_class((0,0,'agentness'));metric_seconds=time.perf_counter()-begin
 expected_ap=read(ORIGINAL_ROOT/'runs/supervised-seed0/detector-results.json')['summary']['agentness'];assert ap==expected_ap
 checks['full_official_detector_agentness_exact']=ap
 gt=[np.array([[0,0,1,1,0]],np.float32)];good=np.array([[0,0,1,1,.9]],np.float32);bad=np.array([[2,2,3,3,.95],[0,0,1,1,.9]],np.float32)
 for det,want in [(good,100),(bad,50)]:
  _,a,_=evaluate.metric.core['evaluate_detections'](gt,[[det]],['event'],iou_thresh=.5);assert float(a[0])==want
 checks['AP_fixtures']=True
 bench=dict(time=time.time(),training_one_epoch_per_condition=measure,training_14_runs_compute_seconds=sum(r['seconds'] for r in measure)*30*7,inference_frames=frames,inference_seconds=elapsed,inference_joint_frames_per_second=frames/elapsed,inference_heads=14,full_inference_compute_seconds=38663*elapsed/frames,full_agentness_metric_seconds=metric_seconds,gpu_peak_gib=gpu_peak,scope='Training-only cache benchmark with disposable seed999 models; inference benchmark uses random fourteen heads on first three locked official videos and computes no semantic AP. Fixed detector-only agentness replays the existing result. Full AP ETA uses prior measured same-evaluator runs, with 16-worker scaling provisional.')
 save(ROOT/'benchmark.json',bench);save(ROOT/'evaluation-preflight.json',dict(passed=True,time=time.time(),checks=checks,evaluation_signature=evaluate.evaluation_signature()));progress('extension-preflight-complete',checks=checks,benchmark=bench)
if __name__=='__main__':main()
