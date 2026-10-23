"""Independent statistics, selection, encoder-state and artifact verification."""
from common import *
import math,statistics,csv

def main():
 verify();assert read(ROOT/'shared-assets-audit.json')['passed'];assert read(ROOT/'study-complete.json')['passed'];out=ROOT/'results';summary=read(out/'summary.json');official=read(INPUTS['frames']);ix=groups(official['labels']['triplet']);data={};weights=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
 for name,h in read(ROOT/'analysis-freeze.json')['files'].items():assert sha(name)==h
 for seed in [0,1,2]:
  paired=[]
  for condition in CONDITIONS:
   folder=ROOT/'runs'/f'{condition}-seed{seed}';d=read(folder/'detector-results.json');t=read(folder/'training-complete.json');hist=read(folder/'history.json');assert len(hist)==30 and min(hist,key=lambda v:v['selection_bce'])['epoch']==t['selected_epoch'];assert sha(folder/'best.pt')==t['checkpoint_sha256']==d['signature']['checkpoint_sha256'];assert t['signature']['data_manifest_sha256']==sha(ROOT/'data-manifest.json');assert d['n_frames']==36717 and d['processed_frames']==38663 and d['n_candidates']==6163968;assert d['frame_sha256']==official['frame_sha256'] and d['candidate_sha256']==official['candidate_sha256']
   ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);assert all(torch.isfinite(v).all() for v in ck['model'].values())
   for k,v in ck['model'].items():
    if k.startswith('stage.') and (condition.startswith('frozen') or 'running_' in k or 'num_batches' in k):assert torch.equal(v,weights['layer3.'+k[6:]])
   if condition.startswith('tuned'):assert not torch.equal(ck['model']['stage.0.conv1.weight'],weights['layer3.0.conv1.weight'])
   for group,values in d['per_class_AP'].items():assert all(math.isfinite(v) and 0<=v<=100 for v in values);assert math.isclose(statistics.mean(values),d['summary'][group],abs_tol=1e-10)
   for group,indices in ix.items():assert math.isclose(statistics.mean(d['per_class_AP']['triplet'][i] for i in indices),d['tail'][group]['mAP'],abs_tol=1e-10)
   data[condition,seed]=d;paired.append(t);del ck
  assert len({t['initialization_sha256'] for t in paired})==len({t['order_sha256'] for t in paired})==1
  assert paired[0]['initialization_sha256']==read(ORIGINAL_ROOT/'runs'/f'supervised-seed{seed}'/'training-complete.json')['initialization_sha256']
 contrasts={'primary_tuning_distilled':{'tuned-distilled':1,'frozen-distilled':-1},'tuning_supervised':{'tuned-supervised':1,'frozen-supervised':-1},'kd_frozen':{'frozen-distilled':1,'frozen-supervised':-1},'kd_tuned':{'tuned-distilled':1,'tuned-supervised':-1},'interaction':{'tuned-distilled':1,'tuned-supervised':-1,'frozen-distilled':-1,'frozen-supervised':1}}
 for endpoint,conditions in summary['endpoints'].items():
  get=lambda v:v['summary'][endpoint] if endpoint in HEADS else v['tail'][endpoint]['mAP']
  for c,r in conditions.items():
   v=[get(data[c,s]) for s in [0,1,2]];assert math.isclose(statistics.mean(v),r['mean'],abs_tol=1e-10) and math.isclose(statistics.stdev(v),r['sample_sd'],abs_tol=1e-10)
  for name,coef in contrasts.items():
   v=[sum(w*get(data[c,s]) for c,w in coef.items()) for s in [0,1,2]];r=summary['contrasts'][endpoint][name];assert math.isclose(statistics.mean(v),r['mean'],abs_tol=1e-10) and math.isclose(statistics.stdev(v),r['sample_sd'],abs_tol=1e-10)
   critical=.95*math.sqrt(2/(1-.95**2));half=critical*statistics.stdev(v)/math.sqrt(3);assert np.allclose(r['paired_t95'],[statistics.mean(v)-half,statistics.mean(v)+half],atol=1e-9,rtol=0)
 rows=list(csv.DictReader((out/'per-class.csv').open()));assert len(rows)==184
 for row in rows:
  for name,coef in contrasts.items():assert math.isclose(float(row[name]),sum(w*float(row[c+'_mean']) for c,w in coef.items()),abs_tol=1e-12)
 for video in read(ROOT/'split-manifest.json')['groups']['final']:assert read(ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']==read(ORIGINAL_ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']
 backups=read(ROOT/'monitor/checkpoints.json');assert backups['passed'] and len(backups['checkpoints'])==24
 for e in backups['checkpoints']:assert sha(e['path'])==e['sha256']==sha(e['backup'])
 for name,h in read(out/'reproducibility.json')['results_sha256'].items():assert sha(out/name)==h
 for c,dest in [('tuned-distilled','bundle-seed0'),('frozen-distilled','bundle-frozen-seed0')]:
  bundle=ROOT/dest;manifest=read(bundle/'bundle.json');assert manifest['student_checkpoint_sha256']==sha(ROOT/'runs'/f'{c}-seed0'/'best.pt')
  for name,h in manifest['files_sha256'].items():assert sha(bundle/name)==h
  bw=torch.load(bundle/'backbone.pth',map_location='cpu',weights_only=True);ck=torch.load(ROOT/'runs'/f'{c}-seed0'/'best.pt',map_location='cpu',weights_only=False)
  for k,v in bw.items():assert torch.equal(v,ck['model']['stage.'+k[7:]] if k.startswith('layer3.') else weights[k])
  hw=torch.load(bundle/'student.pt',map_location='cpu',weights_only=True)
  for k,v in hw.items():assert torch.equal(v,ck['model']['head.'+k])
 assert read(ROOT/'runtime-summary.json')['paired_candidates_exact'] and read(ROOT/'runtime-summary.json')['packaged_instrumented_max_abs']<=1e-7
 save(ROOT/'final-audit.json',dict(time=time.time(),passed=True,new_detector_runs=12,independent_statistics=True,initialization_and_order_paired=True,frozen_encoder_and_BN_verified=True,tuned_layer3_changed=True,input_hash_parity=True,all_classes=True,checkpoints_and_backups_verified=24,bundles_verified=2,raw_video_runtime_verified=True));print('FINAL AUDIT PASSED')
if __name__=='__main__':main()
