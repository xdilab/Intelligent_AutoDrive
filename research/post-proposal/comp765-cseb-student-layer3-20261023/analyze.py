"""Prespecified factorial contrasts, all seeds/classes and runtime; no selection."""
from common import *
import csv,statistics,math,subprocess,shutil
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CONTRASTS={'primary_tuning_distilled':{'tuned-distilled':1,'frozen-distilled':-1},'tuning_supervised':{'tuned-supervised':1,'frozen-supervised':-1},'kd_frozen':{'frozen-distilled':1,'frozen-supervised':-1},'kd_tuned':{'tuned-distilled':1,'tuned-supervised':-1},'interaction':{'tuned-distilled':1,'tuned-supervised':-1,'frozen-distilled':-1,'frozen-supervised':1}}
def stats(v):return dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v))
def interval(v):
 h=4.302652729749*statistics.stdev(v)/math.sqrt(3);return dict(**stats(v),paired_deltas=v,paired_t95=[statistics.mean(v)-h,statistics.mean(v)+h],df=2)
def csv_write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 verify();out=ROOT/'results';out.mkdir(exist_ok=True);official=read(INPUTS['frames']);labels=official['labels'];ix=groups(labels['triplet']);data={};records={}
 for seed in [0,1,2]:
  pair=[]
  for condition in CONDITIONS:
   run=f'{condition}-seed{seed}';folder=ROOT/'runs'/run;d=read(folder/'detector-results.json');t=read(folder/'training-complete.json');assert sha(folder/'best.pt')==t['checkpoint_sha256']==d['signature']['checkpoint_sha256'];assert t['signature']['data_manifest_sha256']==sha(ROOT/'data-manifest.json');assert d['n_frames']==36717 and d['n_candidates']==6163968;pair.append(t);data[condition,seed]=d;records[run]=dict(result_sha256=sha(folder/'detector-results.json'),training=t,path=str(folder))
  assert len({t['initialization_sha256'] for t in pair})==len({t['order_sha256'] for t in pair})==1
 for video in read(ROOT/'split-manifest.json')['groups']['final']:assert read(ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']==read(ORIGINAL_ROOT/'evaluation-inputs'/f'{video}.json')['images_sha256']
 for d in data.values():
  for h in ['agentness','agent']:assert d['per_class_AP'][h]==data['frozen-supervised',0]['per_class_AP'][h]
 rows=[];summary={};contrasts={}
 for endpoint in HEADS+list(ix):
  get=lambda r:r['summary'][endpoint] if endpoint in HEADS else r['tail'][endpoint]['mAP'];summary[endpoint]={};contrasts[endpoint]={}
  for condition in CONDITIONS:
   v=[get(data[condition,s]) for s in [0,1,2]];summary[endpoint][condition]=dict(**stats(v),values=v)
   for s,value in enumerate(v):rows.append(dict(endpoint=endpoint,condition=condition,seed=s,value=value,selected_epoch=records[f'{condition}-seed{s}']['training']['selected_epoch']))
  for name,weights in CONTRASTS.items():contrasts[endpoint][name]=interval([sum(w*get(data[c,s]) for c,w in weights.items()) for s in [0,1,2]])
 csv_write(out/'per-seed.csv',rows);csv_write(out/'contrasts.csv',[dict(endpoint=e,contrast=c,delta_mean=r['mean'],delta_sample_sd=r['sample_sd'],ci_low=r['paired_t95'][0],ci_high=r['paired_t95'][1],**{f'seed{s}':v for s,v in enumerate(r['paired_deltas'])}) for e,cs in contrasts.items() for c,r in cs.items()]);classes=[]
 for group in HEADS:
  for j,label in enumerate(['agentness'] if group=='agentness' else labels[group]):
   row=dict(group=group,class_id=j,label=label,gt_count=data['frozen-supervised',0]['gt_counts'][group][j],tail47=group=='triplet' and j in ix['tail47'],deep28=group=='triplet' and j in ix['deep28'],common39=group=='triplet' and j in ix['common39'])
   for c in CONDITIONS:
    values=[data[c,s]['per_class_AP'][group][j] for s in [0,1,2]]
    for s,v in enumerate(values):row[f'{c}_seed{s}']=v
    row[c+'_mean']=statistics.mean(values)
   for name,weights in CONTRASTS.items():row[name]=sum(w*row[c+'_mean'] for c,w in weights.items())
   classes.append(row)
 csv_write(out/'per-class.csv',classes);changes={}
 for contrast in CONTRASTS:
  changes[contrast]={}
  for subset in ['all86']+list(ix):
   items=[r for r in classes if r['group']=='triplet' and (subset=='all86' or r[subset])];changes[contrast][subset]=dict(improve=sum(r[contrast]>0 for r in items),decline=sum(r[contrast]<0 for r in items),tie=sum(r[contrast]==0 for r in items))
 runtime=read(ROOT/'runtime-summary.json');teacher=read(INPUTS['teacher_result']);save(out/'summary.json',dict(time=time.time(),metric='Official detector AP@0.5 (%)',conditions=CONDITIONS,seeds=[0,1,2],endpoints=summary,contrasts=contrasts,class_changes=changes,teacher_triplet=teacher['summary']['triplet'],all_run_records=records,input_parity_verified=True));shutil.copy2(ROOT/'runtime-summary.json',out/'runtime-summary.json')
 csv_write(out/'runtime-stages-seconds.csv',[dict(condition=c,stage=k,**v) for c in ['frozen','tuned'] for k,v in runtime[c]['summary'].items()])
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180});colors=['#858585','#333333','#729eaa','#126582'];fig,axes=plt.subplots(2,3,figsize=(13,6),sharex=True)
 for s in [0,1,2]:
  for c,color in zip(CONDITIONS,colors):
   h=read(ROOT/'runs'/f'{c}-seed{s}'/'history.json');axes[0,s].plot([r['epoch'] for r in h],[r['loss'] for r in h],label=c,color=color);axes[1,s].plot([r['epoch'] for r in h],[r['selection_bce'] for r in h],color=color)
  axes[0,s].set_title(f'Seed {s}');axes[1,s].set_xlabel('Epoch')
 axes[0,0].set_ylabel('Training objective (losses differ)');axes[1,0].set_ylabel('Selection supervised BCE');axes[0,0].legend(frameon=False,fontsize=7);fig.tight_layout();fig.savefig(out/'training-curves.png');fig.savefig(out/'training-curves.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4.5))
 for c,color in zip(CONDITIONS,colors):
  arm=c.split('-')[0];x=runtime[arm]['summary']['total_seconds']['median']*1000;v=summary['triplet'][c];ax.errorbar(x,v['mean'],yerr=v['sample_sd'],fmt='s' if c.endswith('distilled') else 'o',color=color,capsize=4,label=c)
 ax.set_xlabel('Median serial raw-video processing (ms/frame)');ax.set_ylabel('Official triplet mAP (%)');ax.set_ylim(bottom=0);ax.legend(frameon=False);ax.set_title('Same deployed architecture; mean ± seed sample SD');fig.text(.02,.02,'Timing: distilled seed0, 4 warmup + 32 measured, live conf0.25. AP: fixed full candidates.\nTiming differences are not evidence of a structural speedup. Three paired seeds; reused benchmark.',fontsize=8);fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(out/'accuracy-versus-cost.png');fig.savefig(out/'accuracy-versus-cost.pdf');plt.close(fig)
 table=['| Condition | Triplet mean ± SD | Tail47 | Deep28 | Common39 |','|---|---:|---:|---:|---:|']
 for c in CONDITIONS:
  v=summary['triplet'][c];table.append(f"| {c} | {v['mean']:.3f} ± {v['sample_sd']:.3f} | {summary['tail47'][c]['mean']:.3f} | {summary['deep28'][c]['mean']:.3f} | {summary['common39'][c]['mean']:.3f} |")
 ct=['| Triplet contrast | Mean paired delta | 95% paired-t interval |','|---|---:|---:|']
 for c,v in contrasts['triplet'].items():ct.append(f"| {c} | {v['mean']:+.3f} | [{v['paired_t95'][0]:+.3f}, {v['paired_t95'][1]:+.3f}] |")
 d=contrasts['triplet']['primary_tuning_distilled'];tail=changes['primary_tuning_distilled']['tail47'];params=read(ROOT/'preflight.json')['parameters'];rt=runtime['tuned'];rf=runtime['frozen'];ci='includes zero; this interval leaves the sign inconclusive' if d['paired_t95'][0]<=0<=d['paired_t95'][1] else 'excludes zero on this fixed benchmark, with only three seeds'
 report=f"""# ResNet layer3 accuracy follow-up — COMP 765

Completed twelve runs: four conditions × paired seeds0–2. This post-defense follow-up asks whether partial ResNet fine-tuning improves the existing causal student. Historical studies/checkpoints and the working demo are preserved. The protocol was fixed before these new detector results.

## Architecture and controls

Same ResNet50 ImageNetV2 throughlayer3 at448×672, stride16/1024channels;3×3 RoIAlign mean, scene mean,256-unit GRU, geometry8, agentembedding16 and MLP256→184. Semantic parameters {params['semantic_total']:,}; head {params['head']:,}; layer3 {params['layer3']:,}. Frozen arms train only the head; tuned arms additionally train layer3. All BN running statistics remain fixed, including during training; layer3 BN affine parameters train in tuned arms. Headlr1e-3, layer3lr1e-5, AdamW decay.01, gradientclip5. No architecture or data-size change. The deployed network has identical shape and parameter count in all arms.

All four arms start the same ImageNetV2 encoder and per-seed random head; no warmstart. Same64training/12selection videos,2,432current-image steps,32frame full-BPTT windows,30epochs and minimum supervised selection BCE (first tie). Original labels/teacher targets/candidates are read-only and hash-verified. Frozen prefix throughlayer2 is cached losslessly as bfloat16; layer3 uses32-frame batches with fixedBN and pooledhalf→float features. Fresh frozen controls use this same batched feature path. Frame-batching numerical effects are recorded in preflight; historical frozen AP is contextual only.

Supervised independent Bernoulli BCE versus BCE+weight1teacherBCE on outputs1–183; no softmax/temperature. Slot0 is masked from both losses. Detector confidence is applied exactly once; official agent scores come from YOLO. Unannotated frames update state without loss. Teacher targets use current−3..current+4 at10Hz, privileged training supervision; student inference uses only current/past frames. Teacher targets exist only for the intended training inputs.

## Official detector AP@0.5 (%)

{chr(10).join(table)}

{chr(10).join(ct)}

Primary: tuning within the distilled student gives {d['mean']:+.3f} percentage points. Its95% paired-t(df2) interval {ci}. The interval measures training-seed variation conditional on this fixed benchmark; it does not establish unseen-video generalization or equivalence. Secondary contrasts and endpoints are exploratory, with no multiplicity correction. Tail47 mean-class changes for the primary contrast: {tail['improve']} improve, {tail['decline']} decline, {tail['tie']} tie. All184class outputs and all per-seed values are in the CSVs, including negative changes.

Historical verified CSEB teacher triplet mAP {teacher['summary']['triplet']:.3f}; tuned-distilled mean remains {teacher['summary']['triplet']-summary['triplet']['tuned-distilled']['mean']:.3f} points below it. Teacher uses broader training and future context; this is a descriptive gap, not a matched architecture-only comparison. Its prior cross-environment numerical caveat is retained.

Evaluation:198official videos,36,717scoredframes,6,163,968fixedcandidates;38,663causalsteps fromvideo starts, zero state at boundaries and continuous state through gaps.85/86triplets supported; all86 retained. Fixed training-derivedTail47/Deep28/Common39, no validation-frequency redefinition. Same metric/core, candidate and image hashes as original. These are detector metrics, not tiny-pilot proposal AP.

## Runtime and packaged model

Prespecified tuned-distilled seed0 versus fresh frozen-distilled seed0 on GPU0 RTX A6000, same rawMP4 and exact YOLO candidate parity. Fourwarmup+32measured frames, CUDA synchronization at stage boundaries; liveconf.25/imgsz1280/IoU.7/max300 differs from full-candidate AP. Tuned median {rt['summary']['total_seconds']['median']*1000:.2f}ms, p95 {rt['summary']['total_seconds']['p95']*1000:.2f}ms, {rt['frames_per_processing_second']:.2f}frames/s; frozen median {rf['summary']['total_seconds']['median']*1000:.2f}ms, p95 {rf['summary']['total_seconds']['p95']*1000:.2f}ms, {rf['frames_per_processing_second']:.2f}frames/s. Same architecture, so small timing differences must not be interpreted as a structural efficiency gain. Full loading/decode/preprocessing/detector/encoder/head/export, memory and parameter measurements are provided. Serial file-processing throughput is not sensor-to-decision latency or a deployment guarantee.

The tuned bundle contains the selected layer3 weights merged into the portable ResNet, its selected head, detector and standalone source. It runs from rawvideo with external model/cache access blocked; teacher and teacher caches are absent. Evaluator-to-bundle parity is checked within1e-7. Three complete training videos ({sum(r['frames'] for r in runtime['continuous'])}frames) pass causal indexing, finite outputs and bounded post-warmup memory tests. The old live bundle is unchanged; no automatic promotion based on AP.

## Limitations and reproducibility

Official validation was previously used by thesis research and sharedYOLO checkpoint selection. Semantic teacher/student training and selection excludes these198videos, but this is not an independent full-system test. Selection videos belong to teacher historical development. Three seeds,64training videos,32frame training windows and one fixed fine-tuning schedule limit conclusions; no post-result hyperparameter search or selective seed reporting was performed. CUDA RoIAlign backward can be nondeterministic; resume restores optimizer/all RNG and checks numerical tolerance, with observed error in preflight. No deployment-safety claim.

Code/protocol: `{HERE}`. Full runs/logs/checkpoints: `{ROOT}`. Separate bundle: `{ROOT}/bundle-seed0`. [Per-seed](per-seed.csv), [contrasts](contrasts.csv), [per-class](per-class.csv), [curves](training-curves.png), [accuracy/cost](accuracy-versus-cost.png), [runtime](runtime-summary.json), [reproducibility](reproducibility.json). Temporary prefix caches can be regenerated from hashed sources. Historical artifacts, thesis and slides were not modified. No remote push requested.
"""
 (out/'FINDINGS.md').write_text(report)
 save(out/'reproducibility.json',dict(time=time.time(),revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),launch_provenance=read(ROOT/'launch-provenance.json'),code_sha256={f.name:sha(f) for f in HERE.glob('*.py')},freeze=read(ROOT/'freeze.json'),environment=read(ROOT/'environment.json'),data_manifest_sha256=sha(ROOT/'data-manifest.json'),checkpoints=records,runtime_sha256=sha(ROOT/'runtime-summary.json'),bundle=read(ROOT/'bundle-seed0/bundle.json'),results_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='reproducibility.json'}))
 save(ROOT/'study-complete.json',dict(time=time.time(),passed=True,detector_runs=12,pairs=3,factorial_conditions=4,input_parity=True,runtime_complete=True));progress('study-complete',primary_triplet_delta=d['mean'])
if __name__=='__main__':main()
