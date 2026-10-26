"""Independent artifact, all-score, split, rule and unchanged-model verification."""
from assoc_common import *
import csv,subprocess

def main():
 setup();verify();assert read(ROOT/'asset-audit.json')['passed'];split=read(ROOT/'split-manifest.json');assert sha(ROOT/'split-manifest.json')==sha(BASE_ROOT/'split-manifest.json');assert not set(split['groups']['train'])&set(split['groups']['selection']+split['groups']['final']);assert not set(split['groups']['selection'])&set(split['groups']['final'])
 receipt=read(SCRATCH/'inference.json');assert receipt['signature']==signature();assert set(receipt['videos'])==set(split['groups']['final']);images=0
 for video in split['groups']['final']:
  actual=read(ROOT/'evaluation-inputs'/f'{video}.json');source=read(BASE_ROOT/'evaluation-inputs'/f'{video}.json');assert actual['images_sha256']==source['images_sha256'];assert actual['processed_frames']==split['final_last_frame'][video] and actual['reset_at']==1;images+=actual['processed_frames']
 assert images==38663
 rules=read(ROOT/'rules.json');weights=np.array(rules['weights']);assert weights.shape==(135,135) and np.isfinite(weights).all() and np.max(np.abs(weights))<=1
 csvrows=list(csv.DictReader((ROOT/'rules.csv').open()));assert len(csvrows)==int((weights!=0).sum())==2537
 for row in csvrows:
  assert int(row['joint_actor_frames'])>=5 and int(row['joint_videos'])>=3
  a,b=row['antecedent_on_other_actor'],row['consequent_on_target'];assert a!=b and not a.startswith(b+'-') and not b.startswith(a+'-')
 records=[];score_hashes={};all_diagnostics=[]
 for seed in SEEDS:
  path=checkpoint(seed);h=sha(path);assert h==read(path.parent/'training-complete.json')['checkpoint_sha256'];assert h==sha(BASE_ROOT/'verified-checkpoint-backup'/path.parent.name/'best.pt');records.append(dict(seed=seed,checkpoint=str(path),sha256=h))
  base=np.load(SCRATCH/f'baseline-seed{seed}.npy',mmap_mode='r');assert base.shape==(6163968,184);checks={c:dict(changed_rows=0,changed_event_scores=0,max_abs_score_change=0.,event_changed_counts=np.zeros(135,dtype=np.int64)) for c in ['association','frequency']}
  other={c:np.load(SCRATCH/f'{c}-seed{seed}.npy',mmap_mode='r') for c in checks}
  for start in range(0,len(base),100000):
   b=base[start:start+100000];assert np.isfinite(b).all() and b.min()>=0 and b.max()<=1
   for c,a in other.items():
    v=a[start:start+100000];assert np.isfinite(v).all() and v.min()>=0 and v.max()<=1;assert np.array_equal(v[:,:49],b[:,:49]);d=v[:,49:]-b[:,49:];changed=d!=0;checks[c]['changed_rows']+=int(changed.any(1).sum());checks[c]['changed_event_scores']+=int(changed.sum());checks[c]['max_abs_score_change']=max(checks[c]['max_abs_score_change'],float(np.max(np.abs(d))));checks[c]['event_changed_counts']+=changed.sum(0)
  for c,d in checks.items():d['event_changed_counts']=d['event_changed_counts'].tolist();all_diagnostics.append(dict(seed=seed,condition=c,**d))
  for c in CONDITIONS:
   file=SCRATCH/f'{c}-seed{seed}.npy';score_hashes[str(file)]=sha(file);r=read(ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json');assert r['signature']==dict(signature(),checkpoint_sha256=h);assert r['n_frames']==36717 and r['n_candidates']==6163968
   if c=='baseline':
    historical=read(path.parent/'detector-results.json');assert max(abs(x-y) for head in HEADS for x,y in zip(r['per_class_AP'][head],historical['per_class_AP'][head]))<1e-6
   assert all(np.isfinite(x) for row in r['per_class_AP'].values() for x in row)
  progress('audit-scores-complete',seed=seed)
 runtime=read(ROOT/'runtime-summary.json');assert runtime['checkpoint_bundle_tensor_exact'] and runtime['shared_correction_code_exact'] and runtime['no_teacher_or_feature_assets'] and runtime['memory_bounds_passed']
 comparison=read(ROOT/'analysis/comparison.json');assert comparison['signature']==signature();assert len(list(csv.DictReader((ROOT/'analysis/per-class.csv').open())))==135
 save(ROOT/'score-diagnostics.json',dict(time=time.time(),diagnostics=all_diagnostics,score_hashes=score_hashes,primitive_scores_bit_identical=True))
 save(ROOT/'final-audit.json',dict(passed=True,time=time.time(),protocol_frozen_before_official=True,split_identity=True,training_selection_final_disjoint=True,images_verified=images,source_checkpoints_unchanged=records,all9score_arrays_finite_bounded=True,primitive_scores_bit_identical=True,baseline_per_class_AP_parity_tolerance=1e-6,all135event_class_rows_reported=True,rules_validated=len(csvrows),runtime_frames=runtime['frames'],runtime_teacher_free=True,code_sha256={str(x):sha(x) for x in HERE.glob('*.py')},score_diagnostics_sha256=sha(ROOT/'score-diagnostics.json'),base_revision=read(ROOT/'provenance.json')['base_revision']))
 progress('final-audit-passed',rules=len(csvrows),images=images)
if __name__=='__main__':main()
