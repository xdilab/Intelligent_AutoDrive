"""Batch scheduler around unchanged frozen numerical inference and official AP."""
from context_common import *
import argparse,shutil,gc

def replay_baselines(e,seeds):
 old={Path(x['path']).name:x['sha256'] for x in read(EXPANSION_ROOT/'prediction-buffer-provenance.json')['files']};rows=[]
 for seed in seeds:
  path=e.SCRATCH/f'baseline-seed{seed}.npy';h=sha(path);assert h==old[f'distilled-seed{seed}.npy'],('baseline whole-score parity',seed,h);src=checkpoint(seed).parent/'detector-results.json';result=read(src);result.update(signature=e.evaluation_signature(seed),condition='baseline',run=f'baseline-seed{seed}',source_result=str(src),source_result_sha256=sha(src),reused_after_exact_score_hash_parity=True,score_sha256=h);save(e.ROOT/'runs'/f'baseline-seed{seed}'/'detector-results.json',result);rows.append(dict(seed=seed,score_sha256=h,source_result_sha256=sha(src)));progress('baseline-replay-verified',seed=seed,batch=batch_name(seeds))
 save(e.ROOT/'baseline-replay.json',dict(passed=True,time=time.time(),records=rows))

def audit_batch(e,seeds):
 split=read(ROOT/'split-manifest.json');receipt=read(e.SCRATCH/'inference.json');assert receipt['signature']==signature() and set(receipt['videos'])==set(split['groups']['final']);images=0
 for video in split['groups']['final']:
  record=read(e.ROOT/'evaluation-inputs'/f'{video}.json');assert record['images_sha256']==read(BASE_ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256'];assert record['reset_at']==1;images+=record['processed_frames']
 assert images==38663;hashes={};diagnostics=[]
 for seed in seeds:
  assert sha(checkpoint(seed))==sha(backup(seed))==read(checkpoint(seed).parent/'training-complete.json')['checkpoint_sha256'];base=np.load(e.SCRATCH/f'baseline-seed{seed}.npy',mmap_mode='r');assert base.shape==(6163968,184)
  for c in CONDITIONS:
   path=e.SCRATCH/f'{c}-seed{seed}.npy';arr=np.load(path,mmap_mode='r');changed=0;maxchange=0.
   for start in range(0,len(base),100000):
    a=arr[start:start+100000];b=base[start:start+100000];assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1;assert np.array_equal(a[:,:49],b[:,:49]);delta=a[:,49:]-b[:,49:];changed+=int((delta!=0).sum());maxchange=max(maxchange,float(np.abs(delta).max()))
   h=sha(path);hashes[str(path)]=h;result=read(e.ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json');assert result['signature']==e.evaluation_signature(seed);assert result['n_frames']==36717 and result['n_candidates']==6163968;assert all(np.isfinite(v) for row in result['per_class_AP'].values() for v in row);save(ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json',dict(result,batch_source=str(e.ROOT),score_sha256=h));diagnostics.append(dict(seed=seed,condition=c,changed_event_scores=changed,max_abs_score_change=maxchange));del arr
  del base;progress('batch-score-audit',batch=batch_name(seeds),seed=seed)
 save(e.ROOT/'score-diagnostics.json',dict(time=time.time(),score_hashes=hashes,diagnostics=diagnostics));save(e.ROOT/'batch-complete.json',dict(passed=True,time=time.time(),seeds=seeds,signature=signature(),score_arrays=len(hashes),images_verified=images,finite_bounded=True,primitive_scores_exact=True,baseline_replay=True,score_diagnostics_sha256=sha(e.ROOT/'score-diagnostics.json')));return hashes

def cleanup(e,hashes):
 files=[]
 for path,h in hashes.items():
  f=Path(path);assert f.parent==e.SCRATCH
  if f.exists():assert f.is_file() and f.stat().st_size==4536680576
  files.append(dict(path=path,sha256=h,bytes=4536680576))
 record=dict(time=time.time(),files=files,total_bytes=sum(x['bytes'] for x in files),removed=False,scope='Only this completed audited batch temporary predictions. Original checkpoints/caches/results preserved.');save(e.ROOT/'prediction-buffer-provenance.json',record);shutil.copyfile(e.SCRATCH/'inference.json',e.ROOT/'scratch-inference-receipt.json') if (e.SCRATCH/'inference.json').exists() else None
 for f in files:Path(f['path']).unlink(missing_ok=True)
 (e.SCRATCH/'inference.json').unlink(missing_ok=True);e.SCRATCH.rmdir();record.update(removed=True,removed_time=time.time());save(e.ROOT/'prediction-buffer-provenance.json',record)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=16);ap.add_argument('--benchmark-videos',type=int);a=ap.parse_args();setup();verify();assert read(ROOT/'preflight.json')['passed'];SCRATCH.mkdir(exist_ok=True)
 for seeds in BATCHES:
  e=evaluator(seeds);name=batch_name(seeds)
  if (e.ROOT/'batch-complete.json').exists():
   assert read(e.ROOT/'batch-complete.json')['passed'] and read(e.ROOT/'batch-complete.json')['signature']==signature()
   if e.SCRATCH.exists():cleanup(e,read(e.ROOT/'score-diagnostics.json')['score_hashes'])
   continue
  required=4536680576*len(seeds)*len(CONDITIONS);existing=sum(f.stat().st_blocks*512 for f in e.SCRATCH.glob('*.npy')) if e.SCRATCH.exists() else 0;assert shutil.disk_usage(SCRATCH).free>=required-existing+1024**3,('scratch budget',name,required,existing)
  official,yr,offset,index=e.legacy.load_official();progress('batch-assets-loaded',batch=name,seeds=seeds,expected_score_GiB=required/2**30);complete=e.infer(official,yr,offset,index,a.benchmark_videos)
  timing_path=e.ROOT/('inference-complete.json' if complete else 'inference-benchmark.json');timing=read(timing_path);timing.update(seeds=seeds,student_heads=len(seeds),score_exports=len(seeds)*len(CONDITIONS),scope=f'Fresh current-image encoding shared by{len(seeds)}student heads,{len(seeds)*len(CONDITIONS)}score exports; fixed detector candidates; detector/MP4 decoding excluded');save(timing_path,timing)
  if a.benchmark_videos:
   shutil.copyfile(e.ROOT/'inference-benchmark.json',ROOT/'inference-benchmark.json');return
  assert complete;del yr,index;replay_baselines(e,seeds);e.metrics(official,offset,a.workers);hashes=audit_batch(e,seeds);e.metric.SCORES=None;e.metric.BOXES=None;e.metric.AGENTS=None;e.metric.GTS=None;gc.collect();cleanup(e,hashes);progress('batch-complete',batch=name,seeds=seeds)
 progress('official-evaluation-complete',total_conditions=70,new_corrected_AP_runs=42,new_baseline_replays=7,original_results_reused=21)
if __name__=='__main__':main()
