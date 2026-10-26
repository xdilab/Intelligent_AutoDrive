"""Independent artifact, all-score, split, rule and unchanged-model verification."""
from assoc_common import *
import csv,subprocess,statistics,math

def main():
 setup();verify();assert read(ROOT/'asset-audit.json')['passed'];assert read(ROOT/'replay-audit.json')['passed'];assert read(ROOT/'pilot-replay.json')['passed'];split=read(ROOT/'split-manifest.json');assert sha(ROOT/'split-manifest.json')==sha(BASE_ROOT/'split-manifest.json');assert not set(split['groups']['train'])&set(split['groups']['selection']+split['groups']['final']);assert not set(split['groups']['selection'])&set(split['groups']['final'])
 receipt=read(SCRATCH/'inference.json');assert receipt['signature']==signature();assert set(receipt['videos'])==set(split['groups']['final']);images=0
 for video in split['groups']['final']:
  actual=read(ROOT/'evaluation-inputs'/f'{video}.json');source=read(BASE_ROOT/'evaluation-inputs'/f'{video}.json');assert actual['images_sha256']==source['images_sha256'];assert actual['processed_frames']==split['final_last_frame'][video] and actual['reset_at']==1;images+=actual['processed_frames']
 assert images==38663
 rules=read(ROOT/'rules.json');weights=np.array(rules['weights']);assert weights.shape==(135,135) and np.isfinite(weights).all() and np.max(np.abs(weights))<=1
 csvrows=list(csv.DictReader((ROOT/'single-rules.csv').open()));assert len(csvrows)==int((weights!=0).sum())==rules['rules']
 for row in csvrows:
  assert int(row['joint_actor_frames'])>=5 and int(row['joint_videos'])>=3
  a,b=row['antecedent'],row['consequent'];assert a!=b and not a.startswith(b+'-') and not b.startswith(a+'-')
 pairs=np.asarray(rules['pair_antecedents']);pw=np.asarray(rules['pair_weights']);assert pw.shape==(len(pairs),135) and np.isfinite(pw).all() and np.max(np.abs(pw))<=1;assert int((pw!=0).sum())==rules['pair_rules'];assert (pw!=0).sum(0).max()<=16
 candidates=list(csv.DictReader((ROOT/'pair-candidates.csv').open()));assert len(candidates)==rules['pair_candidate_rules'];retained=[r for r in candidates if r['retained']=='True'];assert len(retained)==rules['pair_rules'];events=rules['events'];pairmap={tuple(v):i for i,v in enumerate(pairs.tolist())}
 for row in retained:
  assert int(row['joint_actor_frames'])>=5 and int(row['joint_videos'])>=3
  a,b,y=[events.index(row[k]) for k in ['antecedent_a','antecedent_b','consequent']];assert abs(pw[pairmap[a,b],y]-float(row['incremental_effect']))<1e-15
  for x,z in [(a,b),(a,y),(b,y)]:assert x!=z and not events[x].startswith(events[z]+'-') and not events[z].startswith(events[x]+'-')
 # Independently verify candidate residual arithmetic and deterministic per-consequent selection.
 for row in candidates:
  lo=min(float(row['single_a_confidence']),float(row['single_b_confidence']));hi=max(float(row['single_a_confidence']),float(row['single_b_confidence']));pc=float(row['pair_confidence']);assert pc<lo or pc>hi
  def odds(x):
   x=min(max(x,1e-6),1-1e-6);return np.log(x/(1-x))
  expected=float(np.clip(odds(pc)-odds(hi if pc>hi else lo),-1,1));assert abs(expected-float(row['incremental_effect']))<1e-12 and abs(expected)>=np.log(1.1)-1e-12
 for consequent in events:
  pool=[r for r in candidates if r['consequent']==consequent];ranked=sorted(pool,key=lambda r:(-int(r['joint_videos']),-abs(float(r['incremental_effect'])),events.index(r['antecedent_a']),events.index(r['antecedent_b'])))
  expected={(r['antecedent_a'],r['antecedent_b']) for r in ranked[:16]};actual={(r['antecedent_a'],r['antecedent_b']) for r in pool if r['retained']=='True'};assert expected==actual
 records=[];score_hashes={};all_diagnostics=[]
 for seed in SEEDS:
  path=checkpoint(seed);h=sha(path);assert h==read(path.parent/'training-complete.json')['checkpoint_sha256'];assert h==sha(BASE_ROOT/'verified-checkpoint-backup'/path.parent.name/'best.pt');records.append(dict(seed=seed,checkpoint=str(path),sha256=h))
  base=np.load(SCRATCH/f'baseline-seed{seed}.npy',mmap_mode='r');assert base.shape==(6163968,184);checks={c:dict(changed_rows=0,changed_event_scores=0,max_abs_score_change=0.,event_changed_counts=np.zeros(135,dtype=np.int64)) for c in CONDITIONS[1:]}
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
 # Recalculate every reported summary/contrast independently with the Python standard library.
 stat_checks=0
 for condition in CONDITIONS:
  for endpoint in ['triplet','duplex','tail47','deep28','common39']:
   vals=[]
   for seed in SEEDS:
    r=read(ROOT/'runs'/f'{condition}-seed{seed}'/'detector-results.json');vals.append(r['summary'][endpoint] if endpoint in ['triplet','duplex'] else r['tail'][endpoint]['mAP'])
   recorded=comparison['conditions'][condition][endpoint];assert vals==recorded['values'];assert abs(statistics.mean(vals)-recorded['mean'])<1e-12;assert abs(statistics.stdev(vals)-recorded['sample_sd'])<1e-12;stat_checks+=1
 for left,right in read(HERE/'protocol.json')['primary_contrasts']+read(HERE/'protocol.json')['control_contrasts']:
  for endpoint in ['triplet','duplex','tail47','deep28','common39']:
   a=comparison['conditions'][left][endpoint]['values'];b=comparison['conditions'][right][endpoint]['values'];delta=[x-y for x,y in zip(a,b)];mean=statistics.mean(delta);sd=statistics.stdev(delta);half=4.302652729911275*sd/math.sqrt(3);recorded=comparison['contrasts'][left+'-minus-'+right][endpoint];assert delta==recorded['deltas'];assert abs(mean-recorded['mean'])<1e-12 and abs(sd-recorded['sample_sd'])<1e-12;assert max(abs(x-y) for x,y in zip([mean-half,mean+half],recorded['ci95']))<1e-12;stat_checks+=1
 save(ROOT/'score-diagnostics.json',dict(time=time.time(),diagnostics=all_diagnostics,score_hashes=score_hashes,primitive_scores_bit_identical=True))
 save(ROOT/'final-audit.json',dict(passed=True,time=time.time(),protocol_frozen_before_official=True,split_identity=True,training_selection_final_disjoint=True,images_verified=images,source_checkpoints_unchanged=records,all21score_arrays_finite_bounded=True,primitive_scores_bit_identical=True,baseline_per_class_AP_parity_tolerance=1e-6,all135event_class_rows_reported=True,independent_statistical_checks=stat_checks,single_rules_validated=len(csvrows),pair_rules_validated=len(retained),runtime_frames=runtime['frames'],runtime_teacher_free=True,code_sha256={str(x):sha(x) for x in HERE.glob('*.py')},score_diagnostics_sha256=sha(ROOT/'score-diagnostics.json'),base_revision=read(ROOT/'provenance.json')['base_revision']))
 progress('final-audit-passed',single_rules=len(csvrows),pair_rules=len(retained),images=images)
if __name__=='__main__':main()
