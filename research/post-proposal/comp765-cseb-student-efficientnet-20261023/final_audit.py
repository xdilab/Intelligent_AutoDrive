"""Independent final metrics, immutable inputs and checkpoint audit."""
from common import *
import math,statistics,csv

def main():
 verify();assert read(ROOT/'study-complete.json')['passed'];out=ROOT/'results';summary=read(out/'summary.json');official=read(INPUTS['frames']);ix=groups(official['labels']['triplet']);data={}
 for architecture,base in [('efficientnet',ROOT),('resnet',ORIGINAL_ROOT)]:
  for seed in [0,1,2]:
   pair=[]
   for condition in ['supervised','distilled']:
    folder=base/'runs'/f'{condition}-seed{seed}';d=read(folder/'detector-results.json');t=read(folder/'training-complete.json');h=read(folder/'history.json');assert len(h)==30 and min(h,key=lambda v:v['selection_bce'])['epoch']==t['selected_epoch'];assert sha(folder/'best.pt')==t['checkpoint_sha256']==d['signature']['checkpoint_sha256'];assert d['n_frames']==36717 and d['processed_frames']==38663 and d['n_candidates']==6163968;assert d['frame_sha256']==official['frame_sha256'] and d['candidate_sha256']==official['candidate_sha256']
    for group,values in d['per_class_AP'].items():assert all(math.isfinite(v) and 0<=v<=100 for v in values);assert math.isclose(statistics.mean(values),d['summary'][group],abs_tol=1e-10)
    for group,indices in ix.items():assert math.isclose(statistics.mean(d['per_class_AP']['triplet'][i] for i in indices),d['tail'][group]['mAP'],abs_tol=1e-10)
    data[architecture,condition,seed]=d;pair.append(t)
   assert pair[0]['initialization_sha256']==pair[1]['initialization_sha256'] and pair[0]['order_sha256']==pair[1]['order_sha256']
 for endpoint,architectures in summary['endpoints'].items():
  for architecture,r in architectures.items():
   get=lambda v:v['summary'][endpoint] if endpoint in HEADS else v['tail'][endpoint]['mAP'];a=[get(data[architecture,'supervised',s]) for s in [0,1,2]];b=[get(data[architecture,'distilled',s]) for s in [0,1,2]];d=[y-x for x,y in zip(a,b)]
   for key,values in [('supervised',a),('distilled',b),('paired_delta',d)]:assert math.isclose(statistics.mean(values),r[key]['mean'],abs_tol=1e-10) and math.isclose(statistics.stdev(values),r[key]['sample_sd'],abs_tol=1e-10)
   # Closed-form df2 critical value, independently derived from CDF.
   critical=.95*math.sqrt(2/(1-.95**2));half=critical*statistics.stdev(d)/math.sqrt(3);assert np.allclose(r['paired_t95'],[statistics.mean(d)-half,statistics.mean(d)+half],atol=1e-9,rtol=0)
 rows=list(csv.DictReader((out/'per-class.csv').open()));assert len(rows)==184
 for row in rows:
  for a in ['efficientnet','resnet']:assert math.isclose(float(row[f'{a}_kd_delta_mean']),float(row[f'{a}_distilled_mean'])-float(row[f'{a}_supervised_mean']),abs_tol=1e-12)
 for video in read(ROOT/'split-manifest.json')['groups']['final']:assert read(ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']==read(ORIGINAL_ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']
 backups=read(ROOT/'monitor/checkpoints.json');assert backups['passed'] and len(backups['checkpoints'])==12
 for entry in backups['checkpoints']:assert sha(entry['path'])==entry['sha256']==sha(entry['backup'])
 for name,h in read(out/'reproducibility.json')['results_sha256'].items():assert sha(out/name)==h
 bundle=ROOT/'bundle-seed0'
 for name,h in read(bundle/'bundle.json')['files_sha256'].items():assert sha(bundle/name)==h
 assert read(ROOT/'runtime-summary.json')['paired_candidates_exact'];assert read(ROOT/'runtime-summary.json')['packaged_instrumented_max_abs']<=1e-7
 save(ROOT/'final-audit.json',dict(time=time.time(),passed=True,new_detector_runs=6,original_resnet_runs_verified=6,independent_statistics=True,original_results_unchanged=True,input_hash_parity=True,all_classes=True,checkpoints_and_backups_verified=12,bundle_verified=True,raw_video_runtime_verified=True));print('FINAL AUDIT PASSED')
if __name__=='__main__':main()
