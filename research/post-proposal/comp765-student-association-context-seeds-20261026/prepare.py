"""Verify all ten existing heads and freeze a seed-only extension of the parent rules."""
from context_common import *
from correction import Correction
from single_correction import Correction as Single
import shutil,subprocess,platform,inspect

def main():
 setup();assert not (ROOT/'freeze.json').exists();assert read(CONTEXT_ROOT/'final-audit.json')['passed'];context.verify();assert sha(BASE_ROOT/'data-manifest.json')==sha(EXPANSION_ROOT/'data-manifest.json');assert read(EXPANSION_ROOT/'input-parity.json')['passed'];assert read(EXPANSION_ROOT/'preflight.json')['passed'];manifest=read(BASE_ROOT/'data-manifest.json');splits=read(BASE_ROOT/'split-manifest.json');expanded=read(EXPANSION_ROOT/'split-manifest.json')
 for key in ['groups','start_frames','sequence_length','final_last_frame']:assert splits[key]==expanded[key]
 copied=['rules.json','short-rules.json','single-rules.csv','pair-candidates.csv','split-manifest.json','mining-summary.json','runtime-summary.json','replay-audit.json','adapter-manifest.json','model-inventory.json']
 for name in copied:shutil.copyfile(CONTEXT_ROOT/name,ROOT/name);assert sha(ROOT/name)==sha(CONTEXT_ROOT/name)
 rules=read(ROOT/'rules.json');correct=Correction(rules,'cuda');short=Single(read(ROOT/'short-rules.json'),'cuda');seqrec=manifest['sequences'][0];assert sha(seqrec['path'])==seqrec['sha256'];seq=torch.load(seqrec['path'],map_location='cpu',weights_only=False);ckrecords={}
 for seed in SEEDS:
  path=checkpoint(seed);h=sha(path);done=read(path.parent/'training-complete.json');assert h==done['checkpoint_sha256']==sha(backup(seed));ck=torch.load(path,map_location='cpu',weights_only=False);model=Student().cuda().eval();model.load_state_dict(ck['model']);assert sum(x.numel() for x in model.parameters())==1108568;assert all(torch.isfinite(x).all() for x in model.parameters())
  with torch.inference_mode():
   state=None
   for raw in seq[:4]:
    row=to_gpu(raw);z,state=model(row['roi'],row['scene'],row['boxes'],row['agents'],state);base=correct.scores(z,row['q']);assert torch.equal(base,p.detector_scores(z.sigmoid(),row['q']));parts=correct.components(z,row['boxes'],row['agents'],row['q']);parts['short_single']=short.delta(z,row['boxes'],row['agents'],row['q'],'association');parts['short_frequency']=short.delta(z,row['boxes'],row['agents'],row['q'],'frequency')
    for delta in parts.values():
     score=correct.scores(z,row['q'],delta,1);assert torch.isfinite(score).all() and delta.abs().max()<=1;assert torch.equal(score[:,:49],base[:,:49]);assert torch.equal(correct.scores(z,row['q'],delta,0),base)
   row=to_gpu(seq[0]);a,s1=model(row['roi'],row['scene'],row['boxes'],row['agents']);b,s2=model(row['roi'],row['scene'],row['boxes'],row['agents']);assert torch.equal(a,b) and torch.equal(s1,s2)
  ckrecords[str(seed)]=dict(path=str(path),sha256=h,backup=str(backup(seed)),selected_epoch=done['selected_epoch'],tensor_sha256=tensor_hash(ck['model']));progress('checkpoint-preflight',seed=seed)
 for seeds in BATCHES:
  e=evaluator(seeds);assert inspect.unwrap(e.infer).__code__.co_filename==str(CONTEXT_CODE/'official.py') and e.metrics.__code__.co_filename==str(CONTEXT_CODE/'official.py');assert e.Correction is Correction and e.SEEDS==seeds;e.ROOT.mkdir(parents=True,exist_ok=True)
  for name in ['rules.json','short-rules.json','split-manifest.json']:shutil.copyfile(ROOT/name,e.ROOT/name)
 inputs=read(BASE_ROOT/'provenance.json')['inputs'];assets={}
 for key in ['frames','counts','yolo','gt','backbone','detector','annotation','teacher_partition']:
  entry=inputs[key];assert sha(entry['path'])==entry['sha256'];assets[key]=entry
 paths=[HERE/n for n in ['context_common.py','prepare.py','official.py','protocol.json']]+[CONTEXT_CODE/n for n in ['assoc_common.py','official.py','correction.py','single_correction.py','protocol.json']]+[CONTEXT_ROOT/n for n in ['freeze.json','final-audit.json','score-diagnostics.json','preflight.json','runtime-summary.json','replay-audit.json']]+[ROOT/n for n in copied]+[BASE_ROOT/'data-manifest.json',EXPANSION_ROOT/'data-manifest.json',EXPANSION_ROOT/'input-parity.json',EXPANSION_ROOT/'prediction-buffer-provenance.json',PARENT/'comp765-cseb-student-20261021/common.py',PARENT/'comp765-cseb-student-20261021/evaluate.py',PARENT/'stage56-full/metric.py',PARENT/'stage56-full/baseline-core.py',OLD/'model.py',OLD/'pilot.py',INPUTS['frames'],INPUTS['counts']]+[checkpoint(s) for s in SEEDS]
 paths.extend(CONTEXT_ROOT/'runs'/f'{c}-seed{s}'/'detector-results.json' for s in range(3) for c in CONDITIONS)
 save(ROOT/'preflight.json',dict(passed=True,time=time.time(),checkpoints=ckrecords,checks=dict(all10_checkpoints_backups_shapes_finite=True,all7_score_conditions_finite_bounded=True,primitive_scores_q_unchanged=True,zero_strength_and_state_reset=True,training_manifests_byte_identical=True,original_expansion_input_parity=True,rules_splits_byte_identical_to_parent=True,frozen_numerical_functions_imported_unchanged=True),assets=assets))
 save(ROOT/'freeze.json',dict(time=time.time(),before_new_seeds_corrected_results=True,files={str(path):sha(path) for path in paths}));save(ROOT/'provenance.json',dict(time=time.time(),base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),parent_root=str(CONTEXT_ROOT),parent_final_audit_sha256=sha(CONTEXT_ROOT/'final-audit.json'),additional_checkpoint_root=str(EXPANSION_ROOT),assets=assets,scientific_change='Only seed count0–2 to0–9; same seven conditions, frozen rules/strength/model/evaluator. User requested after original3outcomes.'))
 for seed in range(3):
  for c in CONDITIONS:
   src=CONTEXT_ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json';result=read(src);result.update(signature=dict(signature(),checkpoint_sha256=sha(checkpoint(seed))),parent_result=str(src),parent_result_sha256=sha(src),reused_after_parent_final_audit=True);save(ROOT/'runs'/f'{c}-seed{seed}'/'detector-results.json',result)
 progress('preflight-passed',checkpoints=10,original_results_preserved=21)
if __name__=='__main__':main()
