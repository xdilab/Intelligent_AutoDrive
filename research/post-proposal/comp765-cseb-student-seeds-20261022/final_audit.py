"""Independent arithmetic, lineage, checkpoint and immutable-original checks."""
from common import *
import math,statistics,csv
def main():
 verify();assert read(ROOT/'study-complete.json')['passed'];out=ROOT/'results';summary=read(out/'summary.json');old=read(ROOT/'original-record.json');official=read(INPUTS['frames']);runs={}
 for seed in range(10):
  pair=[]
  for condition in ['supervised','distilled']:
   folder=(ORIGINAL_ROOT if seed<3 else ROOT)/'runs'/f'{condition}-seed{seed}';d=read(folder/'detector-results.json');t=read(folder/'training-complete.json');h=read(folder/'history.json');assert len(h)==30 and min(h,key=lambda r:r['selection_bce'])['epoch']==t['selected_epoch'];assert sha(folder/'best.pt')==d['signature']['checkpoint_sha256']==t['checkpoint_sha256']
   if seed<3:assert sha(folder/'detector-results.json')==old['original_results'][f'{condition}-seed{seed}']
   else:assert d['signature']['evaluator_sha256']==sha(HERE/'evaluate.py')
   assert d['frame_sha256']==official['frame_sha256'] and d['candidate_sha256']==official['candidate_sha256'];assert d['n_frames']==36717 and d['processed_frames']==38663 and d['n_candidates']==6163968
   for key,values in d['per_class_AP'].items():assert all(math.isfinite(v) and 0<=v<=100 for v in values);assert math.isclose(statistics.mean(values),d['summary'][key],abs_tol=1e-10)
   for key,indices in groups(official['labels']['triplet']).items():assert math.isclose(statistics.mean(d['per_class_AP']['triplet'][i] for i in indices),d['tail'][key]['mAP'],abs_tol=1e-10)
   runs[condition,seed]=d;pair.append(t)
  assert pair[0]['initialization_sha256']==pair[1]['initialization_sha256'] and pair[0]['order_sha256']==pair[1]['order_sha256']
 for cohort,endpoints in summary['cohorts'].items():
  for endpoint,r in endpoints.items():
   get=lambda d:d['summary'][endpoint] if endpoint in HEADS else d['tail'][endpoint]['mAP']
   a=[get(runs['supervised',s]) for s in r['seeds']];b=[get(runs['distilled',s]) for s in r['seeds']];d=[y-x for x,y in zip(a,b)]
   for key,values in [('supervised',a),('distilled',b),('paired_delta',d)]:
    assert math.isclose(statistics.mean(values),r[key]['mean'],abs_tol=1e-10) and math.isclose(statistics.stdev(values),r[key]['sample_sd'],abs_tol=1e-10)
   # Independently verify critical quantile using numerical integration of t PDF.
   df=r['df'];t=r['t975'];x=np.linspace(0,t,8193);pdf=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)*(1+x*x/df)**(-(df+1)/2);cdf=.5+t/8192/3*(pdf[0]+pdf[-1]+4*sum(pdf[1:-1:2])+2*sum(pdf[2:-1:2]));assert abs(cdf-.975)<1e-8
   half=t*statistics.stdev(d)/math.sqrt(len(d));assert np.allclose(r['paired_t95'],[statistics.mean(d)-half,statistics.mean(d)+half],atol=1e-10,rtol=0)
 rows=list(csv.DictReader((out/'per-class.csv').open()));assert len(rows)==184
 for row in rows:
  for seed in range(10):assert math.isclose(float(row[f'delta_seed{seed}']),float(row[f'distilled_seed{seed}'])-float(row[f'supervised_seed{seed}']),abs_tol=1e-12)
 backups=read(ROOT/'monitor/checkpoints.json');assert backups['passed'] and len(backups['checkpoints'])==28
 for ck in backups['checkpoints']:assert sha(ck['path'])==ck['sha256']==sha(ck['backup'])
 assert sha(ORIGINAL_ROOT/'results/summary.json')==old['original_summary_sha256'];assert sha(ORIGINAL_ROOT/'freeze.json')==old['original_freeze_sha256']
 for n,h in read(out/'reproducibility.json')['results_sha256'].items():assert sha(out/n)==h
 save(ROOT/'final-audit.json',dict(time=time.time(),passed=True,total_pairs=10,new_pairs=7,new_checkpoints_verified=28,original_results_unchanged=True,independent_mean_sd_interval=True,critical_t_quantiles_numerically_verified=True,all_per_class_results=True,paired_initialization_order=True,cross_validation_launched=False));print('FINAL AUDIT PASSED')
if __name__=='__main__':main()
