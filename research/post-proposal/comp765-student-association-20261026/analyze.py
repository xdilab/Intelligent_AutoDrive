"""Preregistered paired summaries; all event classes retained, no wording/rule tuning."""
from assoc_common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=['triplet','duplex','tail47','deep28','common39']
def value(r,k):return r['summary'][k] if k in ['triplet','duplex'] else r['tail'][k]['mAP']
def stats(xs):
 a=np.asarray(xs,float);return dict(values=a.tolist(),mean=float(a.mean()),sample_sd=float(a.std(ddof=1)))
def paired(xs,ys):
 d=np.asarray(xs)-np.asarray(ys);m=float(d.mean());half=4.302652729911275*float(d.std(ddof=1))/np.sqrt(3)
 return dict(deltas=d.tolist(),mean=m,sample_sd=float(d.std(ddof=1)),ci95=[m-half,m+half],interval='paired t, df2; only3seeds; descriptive exploratory interval')
def main():
 verify();out=ROOT/'analysis';out.mkdir(exist_ok=True);labels=read(BASE_ROOT/'data-manifest.json')['labels'];records={(c,s):read(ROOT/'runs'/f'{c}-seed{s}'/'detector-results.json') for c in CONDITIONS for s in SEEDS}
 assert all(r['signature']==dict(signature(),checkpoint_sha256=sha(checkpoint(s))) for (c,s),r in records.items())
 summary={c:{k:stats([value(records[c,s],k) for s in SEEDS]) for k in METRICS} for c in CONDITIONS};contrasts={}
 for a,b in [('association','baseline'),('frequency','baseline'),('association','frequency')]:contrasts[a+'-minus-'+b]={k:paired(summary[a][k]['values'],summary[b][k]['values']) for k in METRICS}
 subsets=groups(labels['triplet']);perclass=[]
 for group in ['duplex','triplet']:
  for i,label in enumerate(labels[group]):
   row=dict(group=group,class_id=i,label=label,tail47=group=='triplet' and i in subsets['tail47'],deep28=group=='triplet' and i in subsets['deep28'],gt_instances=records['baseline',0]['gt_counts'][group][i])
   for c in CONDITIONS:
    vals=[records[c,s]['per_class_AP'][group][i] for s in SEEDS];row[c+'_mean']=float(np.mean(vals))
    for s,v in zip(SEEDS,vals):row[f'{c}_seed{s}']=v
   row['association_delta']=row['association_mean']-row['baseline_mean'];row['frequency_delta']=row['frequency_mean']-row['baseline_mean'];row['association_minus_frequency']=row['association_mean']-row['frequency_mean'];perclass.append(row)
 with (out/'per-class.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(perclass[0]));w.writeheader();w.writerows(perclass)
 with (out/'per-seed.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['seed','condition']+METRICS);w.writeheader();w.writerows([dict(seed=s,condition=c,**{k:value(records[c,s],k) for k in METRICS}) for s in SEEDS for c in CONDITIONS])
 counts={}
 for group,rows in [('all_triplet',[r for r in perclass if r['group']=='triplet']),('tail47',[r for r in perclass if r['tail47']]),('deep28',[r for r in perclass if r['deep28']]),('common39',[r for r in perclass if r['group']=='triplet' and not r['tail47']]),('duplex',[r for r in perclass if r['group']=='duplex'])]:
  counts[group]={c:dict(improve=sum(r[c+'_delta']>1e-8 for r in rows),decline=sum(r[c+'_delta']< -1e-8 for r in rows),tie=sum(abs(r[c+'_delta'])<=1e-8 for r in rows)) for c in ['association','frequency']}
 rules=read(ROOT/'rules.json');covered=np.any(np.array(rules['weights'])!=0,axis=0)[49:];rulecoverage={k:int(covered[ix].sum()) for k,ix in subsets.items()}
 runtime=read(ROOT/'runtime-summary.json') if (ROOT/'runtime-summary.json').exists() else None
 results=dict(time=time.time(),signature=signature(),metric='Official detector AP@0.5 (%)',conditions=summary,contrasts=contrasts,changed_classes=counts,rule_triplet_coverage=rulecoverage,frames=36717,candidates=6163968,seeds=SEEDS,coverage_triplet_supported=int(sum(x>0 for x in records['baseline',0]['gt_counts']['triplet'])),development=read(ROOT/'development-selection.json')['aggregate'],strengths=read(ROOT/'development-selection.json')['strengths'],runtime=runtime)
 save(out/'comparison.json',results)
 table=['| Condition | Triplet | Duplex | Tail47 | Deep28 | Common39 |','|---|---:|---:|---:|---:|---:|']
 for c in CONDITIONS:table.append('| '+c+' | '+' | '.join(f"{summary[c][k]['mean']:.3f} ± {summary[c][k]['sample_sd']:.3f}" for k in METRICS)+' |')
 delta=contrasts['association-minus-baseline']['triplet'];vsfreq=contrasts['association-minus-frequency']['triplet'];tail=contrasts['association-minus-baseline']['tail47'];direction='increased' if delta['mean']>0 else 'decreased' if delta['mean']<0 else 'did not change'
 report=f'''# Explicit cross-actor associations on the causal student

Completed exploratory ROAD-Waymo COMP 765 pilot, started 2026-10-26. The original frozen-ResNet distilled student and its scene GRU remain unchanged. This tests incremental usefulness of an explicit association correction; it does not identify what the GRU internally represents. No new student training was performed.

The mean official triplet detector mAP {direction} by {delta['mean']:+.3f} percentage points. The paired95%t interval is [{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}]. With only3seeds and a reused validation benchmark, this is preliminary evidence, not a definitive generalization claim.

All values below are percent AP, mean ± sample standard deviation across seeds0,1,2. IoU0.5;36,717scoredframes,6,163,968fixed candidates,198videos;85/86triplet classes have evaluation GT. All86classes remain in the mean.

'''+ '\n'.join(table)+f'''

Association minus the context-free frequency control is {vsfreq['mean']:+.3f} triplet points (95%paired interval [{vsfreq['ci95'][0]:+.3f}, {vsfreq['ci95'][1]:+.3f}]). This control uses the same rule effects with training-marginal antecedent frequencies, rather than the current scene. It is one calibration control, not an exhaustive test of calibration methods.

Tail47 changes by {tail['mean']:+.3f} points. Across the47tail classes, {counts['tail47']['association']['improve']} improve, {counts['tail47']['association']['decline']} decline and {counts['tail47']['association']['tie']} tie in mean AP. Complete duplex and triplet changes are in per-class.csv; individual seed endpoints are in per-seed.csv.

## Frozen experiment

- Same original distilled checkpoints, seeds0,1,2, selected by historical development supervised BCE. Encoder: frozen ImageNetV2 ResNet50 through layer3, shared448x672 input; actor RoI features/geometry/agent embedding plus256-dimensional scene GRU. No teacher is used at inference.
- Rules mined only from ground-truth actors in the same64training-video32frame segments. Each target-agent/video contributes equal weight. Minimum5joint actor-frames and3distinct training videos; shrinkage toward target-agent marginal, bounded log-odds effect. Labels are mapped through master label strings; no child arrays.
- Other actors only, fixed spatial neighborhood and overlap exclusion. Identity/parent-child event implications removed. At inference, context candidates have detector confidence≥0.25, class-agnostic NMS0.5 and cap30; output candidates remain unchanged. Single pass uses original scores, never corrected context feedback.
- Only duplex/triplet logits change; target-agent compatibility enforced. Primitive scores are bit-identical. Final detector confidence is applied once; contextual confidence only weights evidence.
-2,537rules cover48/86triplet and36/49duplex classes; tail coverage={rulecoverage['tail47']}/47, deep={rulecoverage['deep28']}/28, common={rulecoverage['common39']}/39. Sparse coverage limits conclusions about rare events.
- One common strength per condition selected from[0,0.25,0.5,1] by mean development triplet proposal AP on12videos. Both selected1. Development AP and detector AP are different metrics and are not compared directly. No strength/rule changes after official evaluation.
- Full causal prefixes process38,663current-image steps, including unscored frames; state resets only at video boundaries. All current image hashes match the original evaluation. The unchanged official evaluator/GT/candidates/tail manifest are reused. Primitive-group metrics for corrected conditions are inherited only after per-frame exact score equality checks.

## Interpretation and limits

The comparison measures the practical effect of these fixed association rules on an already contextual student. Any gain can combine scene context with score calibration; even an advantage over the frequency control does not establish a unique learned mechanism. A weak or negative result would not prove that the GRU already learned all useful associations. No GRU ablation was added.

The final198-video benchmark has prior research use and exposure in detector checkpoint selection. Student/semantic-teacher training and selection are disjoint from these videos, but this is not an untouched full-system test. Student development videos came from the teacher's historical development pool. Rules were not tuned on final AP. The3existing checkpoints are paired evaluation replicates, not3new training runs.

Ground-truth presence during mining differs from uncertain detector/student evidence at inference. Nearby actors can share a scene without interacting. Observed associations do not establish causation, intention or future events. Short training segments, dependent actor-frames and limited rare-class support constrain generalization. No statistical non-significance is interpreted as equivalence. There is no deployment-safety claim.

## Reproduction and artifacts

Code: {HERE}

Run root: {ROOT}

Protocol: protocol.json. Frozen code/source/rule/selection hashes: run-root freeze.json. Original model hashes/backups and dependency details: provenance.json and preflight.json. All rules: rules.csv and rules.json. Split manifest: split-manifest.json. Per-seed detector files: runs/<condition>-seed<seed>/detector-results.json. Raw-video predictions and latency records: runtime/<video>/.

No thesis, defense slides, original checkpoints or working inference bundle were changed. No training curves are generated because this is an inference-only pilot. Analysis plots summarize every class, paired endpoints and incremental correction cost.
'''
 if runtime:
  report+=f'''\n## Runtime scope\n\nTeacher-free adapter ran{runtime['frames']}sampled frames across{runtime['videos']}complete videos at10Hz sampling, with fresh state per video and no future decoding. Model weights match the evaluated seed0checkpoint exactly; the same correction module is used in evaluation and the adapter. Asset access was restricted to the original standalone bundle's model files; no teacher/features were opened. CPU/GPU retained-memory bounds passed.\n\nMedian correction time: {runtime['summary']['correction']['median_ms']:.3f}ms; p95: {runtime['summary']['correction']['p95_ms']:.3f}ms. Median serial total: {runtime['summary']['total']['median_ms']:.2f}ms; p95: {runtime['summary']['total']['p95_ms']:.2f}ms. Total includes MP4 decoding, production detector, student, correction and compressed JSON export; excludes camera/display/transport. This file-processing measurement is not a claim of end-to-end real-time latency. Runtime uses production candidate filtering; detector AP uses the fixed full-candidate manifest.\n'''
 (out/'FINDINGS.md').write_text(report)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
 fig,axes=plt.subplots(1,3,figsize=(13,4));colors={'baseline':'#333333','association':'#166697','frequency':'#aa6b16'}
 for c in CONDITIONS:axes[0].plot(SEEDS,summary[c]['triplet']['values'],marker='o',label=c,color=colors[c])
 axes[0].set(xlabel='Original student seed',ylabel='Triplet detector mAP (%)',xticks=SEEDS);axes[0].legend()
 rows=[r for r in perclass if r['group']=='triplet'];axes[1].scatter([r['baseline_mean'] for r in rows],[r['association_mean'] for r in rows],c=['#166697' if r['tail47'] else '#777777' for r in rows],s=18);high=max(max(r['baseline_mean'],r['association_mean']) for r in rows)+1;axes[1].plot([0,high],[0,high],color='#999999',linestyle='--');axes[1].set(xlabel='Baseline class AP (%)',ylabel='Association class AP (%)',title='All86triplets; blue=Tail47')
 axes[2].hist([r['association_delta'] for r in rows],bins=20,color='#166697');axes[2].axvline(0,color='black',linewidth=.8);axes[2].set(xlabel='Class AP change (percentage points)',ylabel='Triplet classes',title='All class changes')
 fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=180);fig.savefig(out/'accuracy.pdf');plt.close(fig)
 if runtime:
  fig,ax=plt.subplots(figsize=(5,4));ax.scatter([0,runtime['summary']['correction']['median_ms']],[summary['baseline']['triplet']['mean'],summary['association']['triplet']['mean']],color=['#333333','#166697'])
  for c,x in [('baseline',0),('association',runtime['summary']['correction']['median_ms'])]:ax.annotate(c,(x,summary[c]['triplet']['mean']),xytext=(4,5),textcoords='offset points')
  ax.set(xlabel='Incremental correction median (ms/frame)',ylabel='Triplet detector mAP (%)',title='Incremental postprocessing cost only');ax.margins(.25);fig.tight_layout();fig.savefig(out/'incremental-cost.png',dpi=180);plt.close(fig)
 progress('analysis-complete',triplet_delta=delta['mean'],tail_delta=tail['mean'])
if __name__=='__main__':main()
