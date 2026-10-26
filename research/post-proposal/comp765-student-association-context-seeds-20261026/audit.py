"""Verify all70results, both completed batch receipts and independent aggregate arithmetic."""
from context_common import *
import csv,math,statistics

def main():
 verify();context.verify();assert read(ROOT/'preflight.json')['passed'];assert read(CONTEXT_ROOT/'final-audit.json')['passed'];assert read(ROOT/'replay-audit.json')['passed'];assert sha(ROOT/'rules.json')==sha(CONTEXT_ROOT/'rules.json') and sha(ROOT/'short-rules.json')==sha(CONTEXT_ROOT/'short-rules.json');batches=[]
 for seeds in BATCHES:
  r=ROOT/'batches'/batch_name(seeds)
  for name in ['rules.json','short-rules.json','split-manifest.json']:assert sha(r/name)==sha(ROOT/name)
  done=read(r/'batch-complete.json');assert done['passed'] and done['signature']==signature() and done['seeds']==seeds;assert read(r/'baseline-replay.json')['passed'];cleanup=read(r/'prediction-buffer-provenance.json');assert cleanup['removed'] and len(cleanup['files'])==len(seeds)*7;assert sha(r/'score-diagnostics.json')==done['score_diagnostics_sha256'];batches.append(dict(batch=batch_name(seeds),audit_sha256=sha(r/'batch-complete.json'),score_diagnostics_sha256=sha(r/'score-diagnostics.json'),cleanup_sha256=sha(r/'prediction-buffer-provenance.json')))
 for entry in read(ROOT/'provenance.json')['assets'].values():assert sha(entry['path'])==entry['sha256']
 comparison=read(ROOT/'analysis/comparison.json');assert comparison['signature']==signature();records={};checkpoints=[]
 for seed in SEEDS:
  path=checkpoint(seed);h=sha(path);assert h==sha(backup(seed))==read(path.parent/'training-complete.json')['checkpoint_sha256'];checkpoints.append(dict(seed=seed,path=str(path),sha256=h,backup=str(backup(seed))))
  for c in CONDITIONS:
   result=read(ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json');assert result['signature']==dict(signature(),checkpoint_sha256=h);assert result['n_frames']==36717 and result['n_candidates']==6163968;assert len(result['per_class_AP']['triplet'])==86 and len(result['per_class_AP']['duplex'])==49;assert all(np.isfinite(v) for row in result['per_class_AP'].values() for v in row);records[c,seed]=result
   if seed<3:assert result['parent_result_sha256']==sha(CONTEXT_ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json')
   if c=='baseline':
    old=read(path.parent/'detector-results.json');assert max(abs(x-y) for head in HEADS for x,y in zip(old['per_class_AP'][head],result['per_class_AP'][head]))<1e-6
 first=records['baseline',0]
 for r in records.values():assert r['candidate_sha256']==first['candidate_sha256'] and r['frame_sha256']==first['frame_sha256'] and r['gt_counts']==first['gt_counts']
 nchecks=0;critical={3:4.302652729911275,7:2.4469118511449692,10:2.2621571627409915}
 for cohort in comparison['cohorts'].values():
  seeds=cohort['seeds']
  for c in CONDITIONS:
   for k in ['triplet','duplex','tail47','deep28','common39']:
    values=[records[c,s]['summary'][k] if k in ['triplet','duplex'] else records[c,s]['tail'][k]['mAP'] for s in seeds];expected=cohort['conditions'][c][k];assert expected['values']==values;assert abs(np.mean(values)-expected['mean'])<1e-12 and abs(np.std(values,ddof=1)-expected['sample_sd'])<1e-12;nchecks+=1
  for a,b in read(HERE/'protocol.json')['primary_contrasts']+read(HERE/'protocol.json')['control_contrasts']:
   for k in ['triplet','duplex','tail47','deep28','common39']:
    x=np.array(cohort['conditions'][a][k]['values']);y=np.array(cohort['conditions'][b][k]['values']);d=x-y;mean=float(d.mean());sd=float(d.std(ddof=1));half=critical[len(seeds)]*sd/np.sqrt(len(seeds));expected=cohort['contrasts'][a+'-minus-'+b][k];assert expected['deltas']==d.tolist();assert abs(expected['mean']-mean)<1e-12 and abs(expected['sample_sd']-sd)<1e-12;assert np.max(np.abs(np.array(expected['ci95'])-[mean-half,mean+half]))<1e-12;nchecks+=1
 assert comparison['conditions']==comparison['cohorts']['combined_0_9']['conditions'] and comparison['contrasts']==comparison['cohorts']['combined_0_9']['contrasts'];assert len(list(csv.DictReader((ROOT/'analysis/per-class.csv').open())))==135;assert len(list(csv.DictReader((ROOT/'analysis/per-seed.csv').open())))==70
 rows=list(csv.DictReader((ROOT/'analysis/per-class.csv').open()));labels=read(BASE_ROOT/'data-manifest.json')['labels'];subsets=groups(labels['triplet']);cell_checks=0
 for row in rows:
  g=row['group'];i=int(row['class_id']);assert row['label']==labels[g][i] and int(row['gt_instances'])==first['gt_counts'][g][i]
  for group in ['tail47','deep28']:assert (row[group]=='True')==(g=='triplet' and i in subsets[group])
  for c in CONDITIONS:
   values=[records[c,seed]['per_class_AP'][g][i] for seed in SEEDS];assert abs(float(row[c+'_mean'])-float(np.mean(values)))<1e-10;cell_checks+=1
   for seed,v in zip(SEEDS,values):assert float(row[f'{c}_seed{seed}'])==v;cell_checks+=1
  for a,b in read(HERE/'protocol.json')['primary_contrasts']+read(HERE/'protocol.json')['control_contrasts']:
   delta=np.mean([records[a,seed]['per_class_AP'][g][i]-records[b,seed]['per_class_AP'][g][i] for seed in SEEDS]);assert abs(float(row[a+'-minus-'+b])-delta)<1e-10;cell_checks+=1
 save(ROOT/'final-audit.json',dict(passed=True,time=time.time(),total_seeds=10,total_condition_results=70,additional_corrected_AP_runs=42,additional_baseline_whole_score_replays=7,original_results_preserved=21,checkpoints=checkpoints,batches=batches,statistical_checks=nchecks,per_class_numeric_cells_verified=cell_checks,all135event_classes_reported=True,parent_rules_byte_identical=True,parent_runtime_replay_reused_by_hash=True,no_new_training=True,source_revision=read(ROOT/'provenance.json')['base_revision'],code_sha256={str(p):sha(p) for p in HERE.glob('*.py')}));progress('final-audit-passed',seeds=10,results=70,statistical_checks=nchecks)
if __name__=='__main__':main()
