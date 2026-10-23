"""Causal full-video student inference and unchanged official detector AP."""
from common import *
from training_model import TrainModel,Prefix,features
import argparse,multiprocessing as mp,pickle
from PIL import Image
spec=importlib.util.spec_from_file_location('student_official_metric',PARENT/'stage56-full/metric.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
SCRATCH=Path('/dev/shm/comp765-cseb-student-layer3-20261023')
def one_class(task):
 start=time.perf_counter();result=metric.one_class(task);return (*result,time.perf_counter()-start)
def load_official():
 official=read(INPUTS['frames']);yr=pickle.load(INPUTS['yolo'].open('rb'))['records'];gt=pickle.load(INPUTS['gt'].open('rb'));assert gt['labels']==official['labels']
 metric.BOXES=[];metric.AGENTS=[];metric.GTS=[[] for _ in HEADS];scale=np.array([840,600,840,600],np.float32);candidate=hashlib.sha256();offset=[0];index={}
 for i,stem in enumerate(official['frames']):
  r=yr[stem];video,fid=stem.rsplit('_',1);g=gt['records'][video+'/'+str(int(fid))]['gt'];box=r['boxes_xyxyn'].astype(np.float32)
  for b in [stem.encode(),np.asarray(r['boxes_xyxyn']).tobytes(),np.asarray(r['conf']).tobytes(),np.asarray(r['cls']).tobytes()]:candidate.update(b)
  metric.BOXES.append(box);metric.AGENTS.append(r['cls'].astype(int));offset.append(offset[-1]+len(box));index[(video,int(fid))]=i
  gb=g['boxes'].astype(np.float32)/scale if g is not None else np.empty((0,4),np.float32)
  metric.GTS[0].append(np.c_[gb,np.zeros(len(gb),np.float32)])
  for j,h in enumerate(HEADS[1:],1):metric.GTS[j].append(metric.core['get_individual_labels'](gb,g[h]).astype(np.float32) if len(gb) else np.empty((0,5),np.float32))
 assert candidate.hexdigest()==official['candidate_sha256'];assert offset[-1]==6163968
 del gt
 return official,yr,np.array(offset),index
def verify_teacher(official):
 t=read(INPUTS['teacher_result']);assert t['signature']['checkpoint_sha256']==sha(p.TEACHER)
 assert t['candidate_sha256']==official['candidate_sha256'] and t['frame_sha256']==official['frame_sha256']
 assert t['signature']['metric_sha256']==sha(PARENT/'stage56-full/metric.py') and t['signature']['core_sha256']==sha(PARENT/'stage56-full/baseline-core.py')
 save(ROOT/'teacher-comparator.json',dict(result=t,source=str(INPUTS['teacher_result']),source_sha256=sha(INPUTS['teacher_result']),verified_exact_checkpoint_candidates_frames_evaluator=True,temporal_caveat='Teacher sees future4frames; historical frozen result reused, not rerun.'))
def evaluation_signature():
 return dict(signature(),provenance_addendum_sha256=sha(HERE/'PROVENANCE-ADDENDUM.md'),evaluator_sha256=sha(HERE/'evaluate.py'),metric_sha256=sha(PARENT/'stage56-full/metric.py'),core_sha256=sha(PARENT/'stage56-full/baseline-core.py'))
@torch.inference_mode()
def step(models,states,backbone,image,box,agents,q):
 x=backbone(image);out={};frozen=None;rows=[dict(boxes=box)]
 for run,m in models.items():
  if not m.tuned and frozen is not None:r=frozen
  else:
   r=features(m.stage,x,rows)[0]
   if not m.tuned:frozen=r
  z,states[run]=m.head(r['roi'],r['scene'],box,agents,states[run]);assert torch.isfinite(states[run]).all()
  scores=p.detector_scores(z.sigmoid(),q);assert torch.isfinite(scores).all();out[run]=scores.cpu().numpy()
 return out
@torch.inference_mode()
def infer(official,yr,offset,index,max_videos=None):
 splits=read(ROOT/'split-manifest.json');SCRATCH.mkdir(exist_ok=True);models={};arrays={};sig=evaluation_signature();cache_intact=True
 for seed in read(HERE/'protocol.json')['seeds']:
  for condition in CONDITIONS:
   run=f'{condition}-seed{seed}';sig[run]=sha(ROOT/'runs'/run/'best.pt')
 for seed in read(HERE/'protocol.json')['seeds']:
  for condition in CONDITIONS:
   run=f'{condition}-seed{seed}';dest=ROOT/'runs'/run;ckpath=dest/'best.pt';done=read(dest/'training-complete.json');assert sha(ckpath)==done['checkpoint_sha256']
   if (dest/'detector-results.json').exists():
    r=read(dest/'detector-results.json');assert r['signature']==dict(evaluation_signature(),checkpoint_sha256=sha(ckpath));continue
   m=TrainModel(condition.startswith('tuned')).cuda().eval();m.load_state_dict(torch.load(ckpath,map_location='cpu',weights_only=False)['model']);models[run]=m
   path=SCRATCH/f'{run}.npy';cache_intact=cache_intact and path.exists();arrays[run]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(int(offset[-1]),184))
 if not models:return
 receipt=SCRATCH/'inference.json';old=read(receipt) if receipt.exists() else {'signature':sig,'videos':[]};assert old['signature']==sig
 # Every completed video is committed only after all six score arrays are flushed.
 if not cache_intact:old['videos']=[]
 complete=set(old['videos']);backbone=Prefix().cuda();started=time.perf_counter();steps=0;videos_this_session=0
 for video in splits['groups']['final']:
  if video in complete:continue
  states={run:None for run in models};hashes={};tvideo=time.perf_counter();last=splits['final_last_frame'][video]
  for fid in range(1,last+1):
   path=image_path(video,fid);image=np.array(Image.open(path).convert('RGB'));hashes[str(fid)]=sha(path);i=index.get((video,fid))
   if i is None:box=torch.empty((0,4),device='cuda');agents=torch.empty(0,dtype=torch.long,device='cuda');q=torch.empty(0,device='cuda')
   else:
    r=yr[official['frames'][i]];box=torch.from_numpy(r['boxes_xyxyn']).cuda().float();agents=torch.from_numpy(r['cls']).cuda().long();q=torch.from_numpy(r['conf']).cuda().float()
   outputs=step(models,states,backbone,image,box,agents,q)
   if i is not None:
    for run,scores in outputs.items():arrays[run][offset[i]:offset[i+1]]=scores
   steps+=1
   if fid%50==0:progress('official-inference',video=video,fid=fid,completed_videos=len(complete),total_videos=198,steps_this_session=steps,elapsed=time.perf_counter()-started)
  for arr in arrays.values():arr.flush()
  save(ROOT/'evaluation-inputs'/f'{video}.json',dict(video=video,images_sha256=hashes,reset_at=1,processed_frames=last,seconds=time.perf_counter()-tvideo))
  complete.add(video);save(receipt,dict(signature=sig,videos=sorted(complete),time=time.time()))
  progress('official-video-complete',video=video,completed_videos=len(complete),total_videos=198,steps_this_session=steps,elapsed=time.perf_counter()-started)
  videos_this_session+=1
  if max_videos and videos_this_session>=max_videos:break
 if len(complete)<198:
  elapsed=time.perf_counter()-started
  save(ROOT/'inference-benchmark.json',dict(signature=sig,processed_frames=steps,videos=videos_this_session,completed_videos=len(complete),seconds=elapsed,frames_per_second=steps/elapsed,scope='All twelve selected heads share frozen current-image prefix; six tuned layer3 branches and one frozen layer3 branch; complete video prefixes, no AP computed',time=time.time()))
  return
 save(ROOT/'inference-complete.json',dict(signature=sig,processed_frames=38663,scored_frames=36717,candidates=int(offset[-1]),seconds_this_session=time.perf_counter()-started,zero_reset_at_every_video=True,continuous_prefix=True,time=time.time(),scratch=str(SCRATCH)))
 del arrays,models,backbone;torch.cuda.empty_cache()
def metrics(official,offset,workers):
 labels=official['labels'];classes=[['agentness']]+[labels[h] for h in HEADS[1:]]
 coverage={h:np.bincount(np.concatenate([g[:,-1] for g in metric.GTS[j]]).astype(int),minlength=len(classes[j])).tolist() for j,h in enumerate(HEADS)}
 assert coverage==read(ROOT/'official-assets-audit.json')['gt_counts']
 for seed in read(HERE/'protocol.json')['seeds']:
  for condition in CONDITIONS:
   run=f'{condition}-seed{seed}';dest=ROOT/'runs'/run;sig=dict(evaluation_signature(),checkpoint_sha256=sha(dest/'best.pt'))
   if (dest/'detector-results.json').exists():assert read(dest/'detector-results.json')['signature']==sig;continue
   scores=np.load(SCRATCH/f'{run}.npy',mmap_mode='r');metric.SCORES=[scores[a:b] for a,b in zip(offset[:-1],offset[1:])]
   partial=dest/'detector-partial.json';values={h:[None]*len(c) for h,c in zip(HEADS,classes)};details={h:[None]*len(c) for h,c in zip(HEADS,classes)};timings=[]
   if partial.exists():
    prior=read(partial);assert prior['signature']==sig;values=prior['values'];details=prior['details'];timings=prior['timings']
   tasks=[(g,c,name) for g,cc in enumerate(classes) for c,name in enumerate(cc) if values[HEADS[g]][c] is None];started=time.perf_counter()
   with mp.get_context('fork').Pool(workers) as pool:
    for g,c,ap,string,seconds in pool.imap_unordered(one_class,tasks):
     values[HEADS[g]][c]=ap;details[HEADS[g]][c]=string;timings.append(dict(group=HEADS[g],class_id=c,seconds=seconds));save(partial,dict(signature=sig,values=values,details=details,timings=timings));progress('official-metrics',run=run,classes_done=sum(v is not None for vv in values.values() for v in vv),total=184,elapsed=time.perf_counter()-started)
   triplet=values['triplet'];tail={k:dict(classes=len(ix),mAP=float(np.mean([triplet[i] for i in ix]))) for k,ix in groups(labels['triplet']).items()};done=read(dest/'training-complete.json')
   save(dest/'detector-results.json',dict(metric='official detector AP@0.5 (%)',signature=sig,seed=seed,condition=condition,run=run,selected_epoch=done['selected_epoch'],n_frames=36717,processed_frames=38663,n_candidates=int(offset[-1]),candidate_sha256=official['candidate_sha256'],frame_sha256=official['frame_sha256'],summary={h:float(np.mean(v)) for h,v in values.items()},tail=tail,per_class_AP=values,per_class_details=details,gt_counts=coverage,seconds=time.perf_counter()-started,time=time.time()))
   metric.SCORES=None;del scores;progress('official-run-complete',run=run,triplet=float(np.mean(triplet)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);ap.add_argument('--benchmark-videos',type=int);a=ap.parse_args();setup();verify()
 assert read(ROOT/'evaluation-preflight.json')['passed']
 official,yr,offset,index=load_official();verify_teacher(official);progress('official-assets-loaded',frames=36717,candidates=int(offset[-1]))
 infer(official,yr,offset,index,a.benchmark_videos)
 if a.benchmark_videos:return
 del yr,index;metrics(official,offset,a.workers);progress('official-evaluation-complete',runs=12)
if __name__=='__main__':main()
