"""Independent arithmetic and artifact audit after all study phases complete."""
from common import *
import csv, statistics, math

def main():
 verify()
 assert read(ROOT/'study-complete.json')['passed']
 summary=read(ROOT/'results/summary.json')['endpoints']
 results={}
 for seed in [0,1,2]:
  pair=[]
  for condition in ['supervised','distilled']:
   folder=ROOT/'runs'/f'{condition}-seed{seed}'
   done=read(folder/'training-complete.json');result=read(folder/'detector-results.json')
   assert done['checkpoint_sha256']==sha(folder/'best.pt')==result['signature']['checkpoint_sha256']
   hist=read(folder/'history.json');assert len(hist)==30
   best=min(hist,key=lambda x:x['selection_bce']);assert best['epoch']==done['selected_epoch']
   assert result['n_frames']==36717 and result['n_candidates']==6163968
   official=read(INPUTS['frames'])
   assert result['frame_sha256']==official['frame_sha256'] and result['candidate_sha256']==official['candidate_sha256']
   assert result['signature']['evaluator_sha256']==sha(HERE/'evaluate.py')
   for group,values in result['per_class_AP'].items():assert math.isclose(statistics.mean(values),result['summary'][group],abs_tol=1e-10)
   for group,indices in groups(official['labels']['triplet']).items():assert math.isclose(statistics.mean(result['per_class_AP']['triplet'][i] for i in indices),result['tail'][group]['mAP'],abs_tol=1e-10)
   assert sum(x>0 for x in result['gt_counts']['triplet'])==85
   for values in result['per_class_AP'].values():assert all(math.isfinite(v) and 0<=v<=100 for v in values)
   results[condition,seed]=result;pair.append(done)
  assert pair[0]['initialization_sha256']==pair[1]['initialization_sha256']
  assert pair[0]['order_sha256']==pair[1]['order_sha256']
 for endpoint,value in summary.items():
  def get(r):return r['summary'][endpoint] if endpoint in HEADS else r['tail'][endpoint]['mAP']
  a=[get(results['supervised',s]) for s in [0,1,2]];b=[get(results['distilled',s]) for s in [0,1,2]];d=[y-x for x,y in zip(a,b)]
  for name,values in [('supervised',a),('distilled',b),('paired_delta',d)]:
   assert math.isclose(statistics.mean(values),value[name]['mean'],abs_tol=1e-10)
   assert math.isclose(statistics.stdev(values),value[name]['sample_sd'],abs_tol=1e-10)
  half=4.302652729749*statistics.stdev(d)/math.sqrt(3)
  assert all(math.isclose(x,y,abs_tol=1e-9) for x,y in zip(value['paired_t95'],[statistics.mean(d)-half,statistics.mean(d)+half]))
 rows=list(csv.DictReader((ROOT/'results/per-class.csv').open()));assert len(rows)==184
 for row in rows:
  for seed in [0,1,2]:
   assert math.isclose(float(row[f'delta_seed{seed}']),float(row[f'distilled_seed{seed}'])-float(row[f'supervised_seed{seed}']),abs_tol=1e-12)
 for group in ['agentness','agent']:
  assert all(r['per_class_AP'][group]==results['supervised',0]['per_class_AP'][group] for r in results.values())
 bundle=read(ROOT/'bundle-seed0/bundle.json')
 for name,h in bundle['files_sha256'].items():assert sha(ROOT/'bundle-seed0'/name)==h
 assert bundle['student_checkpoint_sha256']==results['distilled',0]['signature']['checkpoint_sha256']
 runtime=read(ROOT/'runtime-summary.json');assert runtime['paired_candidates_exact'] and runtime['packaged_instrumented_max_abs']<=1e-7
 for clip in runtime['continuous']:
  r=read(Path(clip['report']));assert r['causal_source_indices_verified'] and r['external_assets_blocked']
  assert clip['memory_bound_passed'] and clip['cpu_memory_bound_passed']
 backup=read(ROOT/'monitor/checkpoints.json');assert backup['passed'] and len(backup['checkpoints'])==12
 for ck in backup['checkpoints']:assert sha(Path(ck['path']))==ck['sha256']==sha(Path(ck['backup']))
 repro=read(ROOT/'results/reproducibility.json')
 for name,h in repro['results_sha256'].items():assert sha(ROOT/'results'/name)==h
 save(ROOT/'final-audit.json',dict(passed=True,time=time.time(),six_results=True,paired_initialization_and_order=True,selection_rule=True,independent_mean_sd_paired_interval=True,all184_perclass=True,fixed_detector_heads_identical=True,bundle_hashes=True,runtime_replay=True,verified_backups=12,protocol_frozen=True))
 print('FINAL AUDIT PASSED',flush=True)
if __name__=='__main__':main()
