"""Fixed-sample seed extension; original, new-only and combined estimates."""
from common import *
import csv,math,statistics,subprocess,shutil
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COHORTS={'original_0_2':list(range(3)),'extension_3_9':list(range(3,10)),'combined_0_9':list(range(10))}
T975={3:4.302652729749,7:2.446911848792,10:2.262157162798}
def stats(values):return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
def csv_write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 verify();out=ROOT/'results';out.mkdir(exist_ok=True);data={};receipts={};original=read(ROOT/'original-record.json')
 for seed in range(10):
  pair=[]
  for condition in ['supervised','distilled']:
   folder=(ORIGINAL_ROOT if seed<3 else ROOT)/'runs'/f'{condition}-seed{seed}';r=read(folder/'detector-results.json');t=read(folder/'training-complete.json')
   assert t['checkpoint_sha256']==sha(folder/'best.pt')==r['signature']['checkpoint_sha256'];assert r['n_frames']==36717 and r['n_candidates']==6163968
   if seed<3:assert sha(folder/'detector-results.json')==original['original_results'][f'{condition}-seed{seed}']
   assert t['signature']['data_manifest_sha256']==sha(ROOT/'data-manifest.json')
   pair.append(t);data[condition,seed]=r;receipts[f'{condition}-seed{seed}']=dict(root=str(folder),result_sha256=sha(folder/'detector-results.json'),training=t)
  assert pair[0]['initialization_sha256']==pair[1]['initialization_sha256'] and pair[0]['order_sha256']==pair[1]['order_sha256']
 official=read(INPUTS['frames']);labels=official['labels'];ix=groups(labels['triplet']);endpoints=HEADS+list(ix);summary={};rows=[]
 for key in endpoints:
  get=lambda r:r['summary'][key] if key in HEADS else r['tail'][key]['mAP']
  for seed in range(10):
   a,b=[get(data[c,seed]) for c in ['supervised','distilled']];rows.append(dict(endpoint=key,seed=seed,block='original' if seed<3 else 'extension',supervised=a,distilled=b,delta=b-a))
 for cohort,seeds in COHORTS.items():
  summary[cohort]={}
  for key in endpoints:
   values=[r for r in rows if r['endpoint']==key and r['seed'] in seeds];a=[r['supervised'] for r in values];b=[r['distilled'] for r in values];d=[r['delta'] for r in values];half=T975[len(seeds)]*statistics.stdev(d)/math.sqrt(len(seeds));mean=statistics.mean(d)
   summary[cohort][key]=dict(n=len(seeds),seeds=seeds,supervised=stats(a),distilled=stats(b),paired_delta=stats(d),paired_deltas=d,paired_t95=[mean-half,mean+half],df=len(seeds)-1,t975=T975[len(seeds)])
 csv_write(out/'per-seed.csv',rows);perclass=[];changes={}
 for group in HEADS:
  names=['agentness'] if group=='agentness' else labels[group]
  for j,label in enumerate(names):
   row=dict(group=group,class_id=j,label=label,gt_count=data['supervised',0]['gt_counts'][group][j],tail47=group=='triplet' and j in ix['tail47'],deep28=group=='triplet' and j in ix['deep28'],common39=group=='triplet' and j in ix['common39'])
   for seed in range(10):
    a,b=[data[c,seed]['per_class_AP'][group][j] for c in ['supervised','distilled']];row.update({f'supervised_seed{seed}':a,f'distilled_seed{seed}':b,f'delta_seed{seed}':b-a})
   for cohort,seeds in COHORTS.items():
    for condition in ['supervised','distilled','delta']:row[f'{cohort}_{condition}_mean']=statistics.mean(row[f'{condition}_seed{s}'] for s in seeds)
   perclass.append(row)
 csv_write(out/'per-class.csv',perclass)
 for cohort in COHORTS:
  changes[cohort]={}
  for subset in ['all86']+list(ix):
   items=[r for r in perclass if r['group']=='triplet' and (subset=='all86' or r[subset])];key=f'{cohort}_delta_mean';changes[cohort][subset]=dict(improve=sum(r[key]>0 for r in items),decline=sum(r[key]<0 for r in items),tie=sum(r[key]==0 for r in items))
 old=read(ORIGINAL_ROOT/'results/summary.json')['endpoints']
 for key in endpoints:
  assert np.allclose(summary['original_0_2'][key]['paired_t95'],old[key]['paired_t95'],atol=1e-9,rtol=0)
 for r in data.values():
  assert r['candidate_sha256']==official['candidate_sha256'] and r['frame_sha256']==official['frame_sha256']
  for group in ['agentness','agent']:assert r['per_class_AP'][group]==data['supervised',0]['per_class_AP'][group]
 save(out/'summary.json',dict(time=time.time(),metric='official detector AP@0.5 (%)',primary='triplet',cohorts=summary,class_changes=changes,interpretation='Result-informed extension: new seven-seed block fixed before outcomes; combined ten-seed interval exploratory, conditional on fixed benchmark. No optional stopping.'))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,ax=plt.subplots(figsize=(8,4));pairs=[r for r in rows if r['endpoint']=='triplet'];ax.bar([r['seed'] for r in pairs],[r['delta'] for r in pairs],color=['#777777' if r['seed']<3 else '#18658b' for r in pairs]);ax.axhline(0,color='black',linewidth=.8);ax.axvline(2.5,color='black',linestyle=':',linewidth=1);ax.set_xticks(range(10));ax.set_xlabel('Paired training seed (0–2 original; 3–9 extension)');ax.set_ylabel('Distilled − supervised triplet AP (points)');ax.set_title('Every paired seed, without selection');fig.tight_layout();fig.savefig(out/'paired-seeds.png');fig.savefig(out/'paired-seeds.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4))
 for j,(cohort,seeds) in enumerate(COHORTS.items()):
  r=summary[cohort]['triplet'];mean=r['paired_delta']['mean'];lo,hi=r['paired_t95'];ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color='#18658b')
 ax.axvline(0,color='#777777',linestyle=':');ax.set_yticks(range(3),['Original 3 seeds','New 7 seeds','Combined 10 seeds']);ax.set_xlabel('Paired triplet AP change and nominal 95% t interval (points)');ax.set_title('Seed uncertainty on the fixed reused validation benchmark');fig.text(.03,.015,'Extension chosen after original results; combined interval is exploratory. These cohorts overlap,\nso their intervals are not independent. No uncertainty over unseen videos is estimated.',fontsize=8);fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(out/'paired-intervals.png');fig.savefig(out/'paired-intervals.pdf');plt.close(fig)
 # All twenty histories, with comparable selection BCE separated from training objectives.
 for field,label,filename in [('selection_bce','Development supervised BCE','selection-curves'),('loss','Training objective (different losses)','training-curves')]:
  fig,axes=plt.subplots(2,5,figsize=(15,6),sharex=True)
  for seed,ax in enumerate(axes.flat):
   for condition,color in [('supervised','#555555'),('distilled','#18658b')]:
    hist=read((ORIGINAL_ROOT if seed<3 else ROOT)/'runs'/f'{condition}-seed{seed}'/'history.json');ax.plot([r['epoch'] for r in hist],[r[field] for r in hist],color=color,label=condition)
   ax.set_title(f'Seed {seed}'+(' (original)' if seed<3 else ''));ax.set_xlabel('Epoch')
  axes[0,0].set_ylabel(label);axes[1,0].set_ylabel(label);axes[0,0].legend(frameon=False,fontsize=8);fig.suptitle(label+' across all ten paired seeds');fig.tight_layout();fig.savefig(out/(filename+'.png'));fig.savefig(out/(filename+'.pdf'));plt.close(fig)
 table=['| Cohort | Pairs | Supervised mean ± SD | Distilled mean ± SD | Paired change | 95% paired-t interval |','|---|---:|---:|---:|---:|---:|']
 for cohort,seeds in COHORTS.items():
  r=summary[cohort]['triplet'];table.append(f"| {cohort} | {len(seeds)} | {r['supervised']['mean']:.3f} ± {r['supervised']['sample_sd']:.3f} | {r['distilled']['mean']:.3f} ± {r['distilled']['sample_sd']:.3f} | {r['paired_delta']['mean']:+.3f} | [{r['paired_t95'][0]:+.3f}, {r['paired_t95'][1]:+.3f}] |")
 extra=['| Endpoint | Supervised (10) | Distilled (10) | Paired change | 95% interval |','|---|---:|---:|---:|---:|']
 for key in ['tail47','deep28','common39','action','loc','duplex']:
  r=summary['combined_0_9'][key];extra.append(f"| {key} | {r['supervised']['mean']:.3f} ± {r['supervised']['sample_sd']:.3f} | {r['distilled']['mean']:.3f} ± {r['distilled']['sample_sd']:.3f} | {r['paired_delta']['mean']:+.3f} | [{r['paired_t95'][0]:+.3f}, {r['paired_t95'][1]:+.3f}] |")
 r=summary['combined_0_9']['triplet'];c=changes['combined_0_9']['tail47'];interpretation='The combined interval includes zero; the direction remains uncertain under this interval.' if r['paired_t95'][0]<=0<=r['paired_t95'][1] else 'The combined nominal interval excludes zero, but this result-informed extension is exploratory and does not create independent test evidence.'
 report=f'''# Fixed-split seed extension — COMP 765

Completed seven additional matched pairs (seeds 3–9), extending the original three to ten. The user requested the extension after seeing the original interval; the seven seeds and stopping rule were frozen before any new training/outcomes. All fourteen new runs use the same 64 training videos, 12 selection videos, 32-frame sequences, frozen ResNet50-layer3 + scene-GRU student, teacher targets, 30 epochs, optimizer, loss weights, initialization/order pairing, checkpoint selection, candidate/frame manifests and official evaluator. Only the seed changes. No wording, architecture, hyperparameter, data or checkpoint tuning used new AP.

## Triplet detector AP@0.5 (%)

{chr(10).join(table)}

{interpretation} The interval measures seed variation conditional on this fixed dataset; it does not quantify generalization to new road videos. The original three-seed result remains unchanged. Original, new-only and combined cohorts are reported separately; the combined cohort overlaps the other two. All seeds are included regardless of outcome. There was no significance-dependent stopping and no confirmatory p-value claim.

## Secondary endpoints

{chr(10).join(extra)}

Tail47 mean-class changes over ten seeds: {c['improve']} improve, {c['decline']} decline, {c['tie']} tie. Every class and seed is in [per-class.csv](per-class.csv) and [per-seed.csv](per-seed.csv); unsupported classes remain in the fixed 86-class mean. Frequency subsets remain training-derived. All evaluations use 36,717 scored frames, 198 videos and 6,163,968 candidates, with 38,663 causal state steps from video starts.

## Limits and cross-validation

The fixed teacher's semantic training/selection excludes these official validation videos, but the shared YOLO detector used all 198 for historical validation/checkpoint selection. Thus this remains a reused benchmark, not an independent system test. More seeds do not repair that exposure or expand training-video coverage. The earlier teacher numerical-parity caveat remains applicable. The original selected seed-0 inference bundle and runtime measurements remain unchanged; no best-of-ten export or new runtime claim is made.

Cross-validation was not launched. The user clarified that five folds are not required. The optional design memo in the code directory explains video-level grouping, inner checkpoint selection, teacher/detector exposure, and why fold scores must not be treated as independent training seeds.

## Records

Original study: `{ORIGINAL_ROOT}`. Extension: `{ROOT}`. Code/protocol: `{HERE}`. [Paired-seed plot](paired-seeds.png), [interval plot](paired-intervals.png), [selection curves](selection-curves.png), [training curves](training-curves.png), and [reproducibility](reproducibility.json). The full original architecture/runtime report remains under the original study; this extension changes only the number of paired seeds. Student t critical values use df2/6/9, consistent with [NIST](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm).
'''
 (out/'FINDINGS.md').write_text(report)
 save(out/'reproducibility.json',dict(time=time.time(),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),code_sha256={x.name:sha(x) for x in HERE.glob('*.py')},protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(ROOT/'freeze.json'),original_record=original,original_environment=read(ROOT/'provenance.json')['environment'],runs=receipts,benchmark=read(ROOT/'benchmark.json'),evaluation_preflight=read(ROOT/'evaluation-preflight.json'),results_sha256={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.name!='reproducibility.json'}))
 save(ROOT/'study-complete.json',dict(time=time.time(),passed=True,new_pairs=7,total_pairs=10,results=str(out/'FINDINGS.md'),cross_validation_launched=False));progress('study-complete',combined_triplet_delta=summary['combined_0_9']['triplet']['paired_delta']['mean'])
if __name__=='__main__':main()
