"""Original full-candidate detector protocol with three stateless score conditions."""
from assoc_common import *
from correction import Correction
from single_correction import Correction as Single
import argparse,multiprocessing as mp
from PIL import Image
metric=legacy.metric
one_class=legacy.one_class

def evaluation_signature(seed=None):
 d=signature()
 if seed is not None:d['checkpoint_sha256']=sha(checkpoint(seed))
 return d

@torch.inference_mode()
def infer(official,yr,offset,index,max_videos=None):
 splits=read(ROOT/'split-manifest.json');SCRATCH.mkdir(exist_ok=True);models={};arrays={};sig=signature();intact=True
 for seed in SEEDS:
  m=Student().cuda().eval();m.load_state_dict(torch.load(checkpoint(seed),map_location='cpu',weights_only=False)['model']);models[seed]=m
  for condition in CONDITIONS:
   run=f'{condition}-seed{seed}';path=SCRATCH/f'{run}.npy';intact=intact and path.exists();arrays[run]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(int(offset[-1]),184))
 receipt=SCRATCH/'inference.json';old=read(receipt) if receipt.exists() else dict(signature=sig,videos=[]);assert old['signature']==sig
 if not intact:old['videos']=[]
 complete=set(old['videos']);backbone=SharedBackbone(p.WEIGHTS).cuda();correct=Correction(read(ROOT/'rules.json'),'cuda');short=Single(read(ROOT/'short-rules.json'),'cuda');started=time.perf_counter();steps=0;count=0;phase_times={k:[] for k in ['image','encoder','heads_and_correction_export','array_write']}
 for video in splits['groups']['final']:
  if video in complete:continue
  states={seed:None for seed in SEEDS};hashes={};tv=time.perf_counter();last=splits['final_last_frame'][video];original_hashes=read(BASE_ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']
  for fid in range(1,last+1):
   t=time.perf_counter();path=image_path(video,fid);image=np.array(Image.open(path).convert('RGB'));h=sha(path);assert h==original_hashes[str(fid)],(video,fid);hashes[str(fid)]=h;i=index.get((video,fid));phase_times['image'].append(time.perf_counter()-t)
   if i is None:box=torch.empty((0,4),device='cuda');agents=torch.empty(0,dtype=torch.long,device='cuda');q=torch.empty(0,device='cuda')
   else:
    r=yr[official['frames'][i]];box=torch.from_numpy(r['boxes_xyxyn']).cuda().float();agents=torch.from_numpy(r['cls']).cuda().long();q=torch.from_numpy(r['conf']).cuda().float()
   t=time.perf_counter();roi,scene=backbone(image,box);roi=roi.half().float();scene=scene.half().float();torch.cuda.synchronize();phase_times['encoder'].append(time.perf_counter()-t);t=time.perf_counter();out={}
   for seed,m in models.items():
    z,states[seed]=m(roi,scene,box,agents,states[seed]);assert torch.isfinite(states[seed]).all();base=correct.scores(z,q)
    assert torch.equal(base,p.detector_scores(z.sigmoid(),q))
    out[f'baseline-seed{seed}']=base.cpu().numpy()
    deltas=correct.components(z,box,agents,q)
    deltas['short_single']=short.delta(z,box,agents,q,'association');deltas['short_frequency']=short.delta(z,box,agents,q,'frequency')
    for condition in CONDITIONS[1:]:
     score=correct.scores(z,q,deltas[condition],1);assert torch.isfinite(score).all();assert torch.equal(score[:,:49],base[:,:49]);out[f'{condition}-seed{seed}']=score.cpu().numpy()
   phase_times['heads_and_correction_export'].append(time.perf_counter()-t);t=time.perf_counter()
   if i is not None:
    for run,score in out.items():arrays[run][offset[i]:offset[i+1]]=score
   phase_times['array_write'].append(time.perf_counter()-t);steps+=1
   if fid%50==0:progress('official-inference',video=video,fid=fid,completed_videos=len(complete),total_videos=198,steps_this_session=steps,elapsed=time.perf_counter()-started)
  for a in arrays.values():a.flush()
  save(ROOT/'evaluation-inputs'/f'{video}.json',dict(video=video,images_sha256=hashes,processed_frames=last,reset_at=1,seconds=time.perf_counter()-tv,original_image_hashes_verified=True))
  complete.add(video);save(receipt,dict(signature=sig,videos=sorted(complete),time=time.time()));count+=1
  progress('official-video-complete',video=video,completed_videos=len(complete),steps_this_session=steps,elapsed=time.perf_counter()-started)
  if max_videos and count>=max_videos:break
 elapsed=time.perf_counter()-started
 timing={k:dict(median_ms=1000*float(np.median(v)) if v else 0,p95_ms=1000*float(np.percentile(v,95)) if v else 0,total_seconds=float(sum(v))) for k,v in phase_times.items()}
 report=dict(signature=sig,processed_frames=steps,videos=count,completed_videos=len(complete),seconds=elapsed,frames_per_second=steps/elapsed,phase_timing=timing,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope='Fresh raw current-image encoding shared by3student heads,21score exports; fixed detector candidates; detector/MP4 decoding excluded; not deployment latency',time=time.time())
 save(ROOT/('inference-complete.json' if len(complete)==198 else 'inference-benchmark.json'),report)
 del arrays,models,backbone;torch.cuda.empty_cache()
 return len(complete)==198

def replay_pilot():
 records=[];hashes=read(PILOT_ROOT/'score-diagnostics.json')['score_hashes']
 for seed in SEEDS:
  for condition,oldcondition in [('baseline','baseline'),('short_single','association'),('short_frequency','frequency')]:
   run=f'{condition}-seed{seed}';path=SCRATCH/f'{run}.npy';h=sha(path);key=str(Path('/dev/shm')/PILOT_CODE.name/f'{oldcondition}-seed{seed}.npy');assert h==hashes[key],('pilot full-score parity',run,h,hashes[key]);source=PILOT_ROOT/'runs'/f'{oldcondition}-seed{seed}'/'detector-results.json';result=read(source);result.update(signature=evaluation_signature(seed),condition=condition,run=run,reused_after_exact_score_hash_parity=True,source_result=str(source),source_result_sha256=sha(source),score_sha256=h);save(ROOT/'runs'/run/'detector-results.json',result);records.append(dict(run=run,score_sha256=h,source_result_sha256=sha(source)));progress('pilot-score-replay-verified',run=run)
 save(ROOT/'pilot-replay.json',dict(passed=True,time=time.time(),records=records))

def metrics(official,offset,workers):
 labels=official['labels'];classes=[['agentness']]+[labels[h] for h in HEADS[1:]]
 coverage={h:np.bincount(np.concatenate([g[:,-1] for g in metric.GTS[j]]).astype(int),minlength=len(classes[j])).tolist() for j,h in enumerate(HEADS)}
 assert coverage==read(BASE_ROOT/'official-assets-audit.json')['gt_counts']
 for seed in SEEDS:
  for condition in CONDITIONS:
   run=f'{condition}-seed{seed}';dest=ROOT/'runs'/run;sig=evaluation_signature(seed)
   if (dest/'detector-results.json').exists():assert read(dest/'detector-results.json')['signature']==sig;continue
   scores=np.load(SCRATCH/f'{run}.npy',mmap_mode='r');metric.SCORES=[scores[a:b] for a,b in zip(offset[:-1],offset[1:])]
   partial=dest/'detector-partial.json';values={h:[None]*len(c) for h,c in zip(HEADS,classes)};details={h:[None]*len(c) for h,c in zip(HEADS,classes)};timings=[]
   if condition!='baseline':
    baseline=read(ROOT/'runs'/f'baseline-seed{seed}'/'detector-results.json')
    # Primitive score equality is asserted per frame above; evaluator's agent scores use identical detector IDs/q.
    for h in HEADS[:4]:values[h]=baseline['per_class_AP'][h];details[h]=baseline['per_class_details'][h]
   if partial.exists():
    prior=read(partial);assert prior['signature']==sig;values=prior['values'];details=prior['details'];timings=prior['timings']
   tasks=[(g,c,name) for g,cc in enumerate(classes) for c,name in enumerate(cc) if values[HEADS[g]][c] is None];started=time.perf_counter()
   with mp.get_context('fork').Pool(workers) as pool:
    for g,c,ap,string,seconds in pool.imap_unordered(one_class,tasks):
     values[HEADS[g]][c]=ap;details[HEADS[g]][c]=string;timings.append(dict(group=HEADS[g],class_id=c,seconds=seconds));save(partial,dict(signature=sig,values=values,details=details,timings=timings));progress('official-metrics',run=run,classes_done=sum(v is not None for vv in values.values() for v in vv),total=184,elapsed=time.perf_counter()-started)
   if condition=='baseline':
    prior=read(checkpoint(seed).parent/'detector-results.json');error=max(abs(a-b) for h in HEADS for a,b in zip(values[h],prior['per_class_AP'][h]));assert error<1e-6,('baseline parity',seed,error)
   else:error=None
   triplet=values['triplet'];tail={k:dict(classes=len(ix),mAP=float(np.mean([triplet[i] for i in ix]))) for k,ix in groups(labels['triplet']).items()}
   save(dest/'detector-results.json',dict(metric='official detector AP@0.5 (%)',signature=sig,seed=seed,condition=condition,run=run,strength=0 if condition=='baseline' else 1,n_frames=36717,processed_frames=38663,n_candidates=int(offset[-1]),candidate_sha256=official['candidate_sha256'],frame_sha256=official['frame_sha256'],summary={h:float(np.mean(v)) for h,v in values.items()},tail=tail,per_class_AP=values,per_class_details=details,gt_counts=coverage,baseline_max_AP_error=error,unchanged_groups_inherited_from_exact_baseline=HEADS[:4] if condition!='baseline' else [],seconds=time.perf_counter()-started,time=time.time()))
   metric.SCORES=None;del scores;progress('official-run-complete',run=run,triplet=float(np.mean(triplet)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);ap.add_argument('--benchmark-videos',type=int);a=ap.parse_args();setup();verify();assert read(ROOT/'preflight.json')['passed']
 official,yr,offset,index=legacy.load_official();progress('official-assets-loaded',frames=36717,candidates=int(offset[-1]))
 complete=infer(official,yr,offset,index,a.benchmark_videos)
 if a.benchmark_videos:return
 assert complete;del yr,index;replay_pilot();metrics(official,offset,a.workers);progress('official-evaluation-complete',runs=21)
if __name__=='__main__':main()
