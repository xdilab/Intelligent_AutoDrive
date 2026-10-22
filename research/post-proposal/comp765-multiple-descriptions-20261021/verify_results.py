import sys,math,json,time,csv,subprocess
from pathlib import Path
import numpy as np
import torch
from common import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);root=ap.parse_args().root
assert read(root/'job.json')['status']=='complete'
verify_freeze(root)
manifest=read(INPUTS['frames']);labels=manifest['labels']['triplet'];indices=groups(labels)
report=read(root/'analysis/comparison.json');completed=[];all_values={}
for seed in [0,1,2]:
 pair=[];histories=[]
 for condition in ['single','multi']:
  dest=root/'runs'/f'{condition}-seed{seed}';result=read(dest/'detector-results.json');done=read(dest/'training-complete.json')
  assert result['selected_epoch']==10 and result['n_frames']==36717
  assert result['frame_sha256']==manifest['frame_sha256'] and result['candidate_sha256']==manifest['candidate_sha256']
  assert sha(dest/'final.pt')==result['signature']['checkpoint_sha256']==done['checkpoint_sha256']
  final=torch.load(dest/'final.pt',map_location='cpu',weights_only=False);last=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
  assert final['epoch']==last['epoch']==10 and last['steps']==2290
  assert all(torch.equal(v,last['state'][k]) for k,v in final['state'].items())
  proto=torch.load(root/f'prototypes-{condition}.pt',weights_only=False)['embeds']
  assert torch.equal(final['state']['P'],torch.nn.functional.normalize(proto.float(),dim=-1))
  assert all(torch.isfinite(v).all() for v in final['state'].values())
  for group,values in result['per_class_AP'].items():
   assert np.isfinite(values).all() and abs(float(np.mean(values))-result['summary'][group])<1e-12
  for name,ix in indices.items():assert abs(float(np.mean(np.array(result['per_class_AP']['triplet'])[ix]))-result['tail'][name]['mAP'])<1e-12
  history=read(dest/'epochs.json');assert [r['epoch'] for r in history]==list(range(1,11));assert all(math.isfinite(r['loss']) for r in history)
  histories.append([r['permutation_sha256'] for r in history]);pair.append(read(dest/'initialization.json'))
  all_values[condition,seed]=result
  completed.append(dict(run=dest.name,selected_epoch=10,steps=2290,checkpoint_sha256=sha(dest/'final.pt'),detector_result_sha256=sha(dest/'detector-results.json')))
 assert histories[0]==histories[1] and pair[0]['trainable_sha256']==pair[1]['trainable_sha256']
 assert pair[0]['parameters']==pair[1]['parameters']==524985
reference=all_values['single',0]['per_class_AP']
assert all(r['per_class_AP']['agentness']==reference['agentness'] and r['per_class_AP']['agent']==reference['agent'] for r in all_values.values())
for name in ['tail47','triplet','deep28','common39']:
 a=[all_values['single',s]['tail'][name]['mAP'] if name in indices else all_values['single',s]['summary'][name] for s in [0,1,2]]
 b=[all_values['multi',s]['tail'][name]['mAP'] if name in indices else all_values['multi',s]['summary'][name] for s in [0,1,2]]
 d=[bb-aa for aa,bb in zip(a,b)];mean=sum(d)/3;sd=math.sqrt(sum((v-mean)**2 for v in d)/2);half=4.302652729696142*sd/math.sqrt(3)
 got=report['metrics'][name]['paired_delta'];assert abs(got['mean']-mean)<1e-12 and np.allclose(got['ci95'],[mean-half,mean+half],rtol=0,atol=1e-10)
with (root/'analysis/per-class-triplet.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==86 and [r['label'] for r in rows]==labels
for row in rows:
 i=int(row['class_id']);expected=np.mean([all_values['multi',s]['per_class_AP']['triplet'][i]-all_values['single',s]['per_class_AP']['triplet'][i] for s in [0,1,2]])
 assert abs(float(row['paired_delta_mean'])-expected)<1e-12
save(root/'final-audit.json',dict(passed=True,time=time.time(),frozen_files_unchanged=True,original_input_metadata_unchanged=True,selected_epoch=10,paired_initializations_and_all_epoch_orders_equal=True,agent_and_agentness_AP_identical_across_all_runs=True,independent_summary_and_interval_checks=True,all86_csv_rows_verified=True,runs=completed))
freeze=read(root/'freeze.json')
save(root/'reproducibility.json',dict(study=read(HERE/'protocol.json')['study'],training_code_revision=freeze['code_revision'],asset_audit_code_revision=read(root/'provenance.json')['git_revision'],revision_note='Asset audit preceded the implementation commit; freeze records the exact code revision used for all primary runs.',freeze_sha256=sha(root/'freeze.json'),protocol_sha256=sha(HERE/'protocol.json'),dependencies=read(root/'provenance.json')['dependencies'],seeds=[0,1,2],selection='Final epoch10; no validation/development selection',frames=36717,training_rows=3743456,validation_candidates=6163968,frame_sha256=manifest['frame_sha256'],candidate_sha256=manifest['candidate_sha256'],metric='Detector frame AP@0.5, percentages',runtime_note='Float32 batched GPU head inference in both fresh conditions; same score assembly and unchanged published NumPy AP functions. Older historical evaluator inferred per-frame on CPU; historical AP is not the primary baseline.',inputs=read(root/'provenance.json')['inputs'],runs=completed,scope='Separate COMP765 coursework; thesis, slides, original results and raw assets unchanged'))
print(json.dumps({'passed':True,'runs':6,'primary':report['metrics']['tail47'],'class_counts':report['class_counts']},indent=2))
