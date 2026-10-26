"""All preregistered contrasts, seeds and event classes; no result-driven tuning."""
from assoc_common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METRICS=['triplet','duplex','tail47','deep28','common39']
def value(r,k):return r['summary'][k] if k in ['triplet','duplex'] else r['tail'][k]['mAP']
def stats(xs):
 a=np.array(xs);return dict(values=a.tolist(),mean=float(a.mean()),sample_sd=float(a.std(ddof=1)))
def paired(a,b):
 d=np.array(a)-np.array(b);half=4.302652729911275*d.std(ddof=1)/np.sqrt(3);m=float(d.mean());return dict(deltas=d.tolist(),mean=m,sample_sd=float(d.std(ddof=1)),ci95=[float(m-half),float(m+half)],interval='Paired t df2;3seeds; exploratory multiple contrasts; no multiplicity adjustment')
def main():
 verify();out=ROOT/'analysis';out.mkdir(exist_ok=True);labels=read(BASE_ROOT/'data-manifest.json')['labels'];records={(c,s):read(ROOT/'runs'/f'{c}-seed{s}'/'detector-results.json') for c in CONDITIONS for s in SEEDS};assert all(r['signature']==dict(signature(),checkpoint_sha256=sha(checkpoint(s))) for (c,s),r in records.items());summary={c:{k:stats([value(records[c,s],k) for s in SEEDS]) for k in METRICS} for c in CONDITIONS};protocol=read(HERE/'protocol.json');contrasts={a+'-minus-'+b:{k:paired(summary[a][k]['values'],summary[b][k]['values']) for k in METRICS} for a,b in protocol['primary_contrasts']+protocol['control_contrasts']};subsets=groups(labels['triplet']);rows=[]
 for g in ['duplex','triplet']:
  for i,label in enumerate(labels[g]):
   r=dict(group=g,class_id=i,label=label,tail47=g=='triplet' and i in subsets['tail47'],deep28=g=='triplet' and i in subsets['deep28'],gt_instances=records['baseline',0]['gt_counts'][g][i])
   for c in CONDITIONS:
    v=[records[c,s]['per_class_AP'][g][i] for s in SEEDS];r[c+'_mean']=float(np.mean(v))
    for s,x in zip(SEEDS,v):r[f'{c}_seed{s}']=x
   for a,b in protocol['primary_contrasts']+protocol['control_contrasts']:r[a+'-minus-'+b]=r[a+'_mean']-r[b+'_mean']
   rows.append(r)
 with (out/'per-class.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (out/'per-seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['seed','condition']+METRICS);w.writeheader();w.writerows([dict(seed=s,condition=c,**{k:value(records[c,s],k) for k in METRICS}) for s in SEEDS for c in CONDITIONS])
 changes={}
 for g,subset in [('triplet',[r for r in rows if r['group']=='triplet']),('tail47',[r for r in rows if r['tail47']]),('deep28',[r for r in rows if r['deep28']]),('common39',[r for r in rows if r['group']=='triplet' and not r['tail47']]),('duplex',[r for r in rows if r['group']=='duplex'])]:changes[g]={k:dict(improve=sum(r[k]>1e-8 for r in subset),decline=sum(r[k]<-1e-8 for r in subset),tie=sum(abs(r[k])<=1e-8 for r in subset)) for k in contrasts}
 full=read(ROOT/'rules.json');short=read(ROOT/'short-rules.json');coverage={}
 for name,w in [('short_single',short['weights']),('full_single',full['weights']),('pair_increment',full['pair_weights'])]:
  covered=np.array(w)!=0;anyc=covered.any(0);coverage[name]=dict(rules=int(covered.sum()),duplex=int(anyc[:49].sum()),triplet=int(anyc[49:].sum()),**{k:int(anyc[49:][ix].sum()) for k,ix in subsets.items()})
 runtime=read(ROOT/'runtime-summary.json') if (ROOT/'runtime-summary.json').exists() else None;mining=read(ROOT/'mining-summary.json');save(out/'comparison.json',dict(time=time.time(),signature=signature(),metric='Official detector AP@0.5 (%)',seeds=SEEDS,conditions=summary,contrasts=contrasts,changed_classes=changes,rule_coverage=coverage,mining=mining,runtime=runtime,frames=36717,videos=198,candidates=6163968,triplets_supported=85))
 table=['| Condition | Triplet | Duplex | Tail47 | Deep28 | Common39 |','|---|---:|---:|---:|---:|---:|']
 for c in CONDITIONS:table.append('| '+c+' | '+' | '.join(f"{summary[c][k]['mean']:.3f} ± {summary[c][k]['sample_sd']:.3f}" for k in METRICS)+' |')
 ct=['| Contrast | Triplet change (pp) | 95% paired-t interval |','|---|---:|---:|']
 for k,v in contrasts.items():d=v['triplet'];ct.append(f"| {k} | {d['mean']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
 report='''# Full-video mining and two-event association rules

Completed exploratory COMP765 extension, 2026-10-26. This is an inference-time extension of the existing teacher–student study, separate from the thesis. The unchanged frozen ResNet50 through layer3, shared448×672 visual encoder, actor RoI/geometry/agent features and256-dimensional scene GRU use the original distilled seeds0,1,2. No new student training was performed.

The substantive interventions are additional training-only mining frames and two-event conjunction rules. The original pilot and its historical artifacts remain immutable. The seed-only ten-seed draft was not launched and is superseded by this methodological extension.

'''+ '\n'.join(table)+ '\n\n'+ '\n'.join(ct)+f'''

All metrics are official detector AP@IoU0.5 in percent. Mean ± sample standard deviation across3paired checkpoints. Evaluation:198videos,36,717scoredframes,38,663causal prefix steps,6,163,968fixed candidates,85/86triplets supported; all86remain in the mean. Full per-seed and all135event-class results are in per-seed.csv and per-class.csv.

The confidence intervals have only2degrees of freedom. They summarize checkpoint-seed variability on this reused benchmark, not uncertainty from a new video sample; multiple contrasts are exploratory and unadjusted. Non-significance is not equivalence.

## Controlled design

- Data scale: full_single−short_single holds architecture, checkpoints, inference/evaluator, rule criteria and strength1 fixed; it changes the training annotation coverage.
- Rule complexity: full_pair−full_single adds only the predeclared residual two-event rule contribution. It retains the same expanded mining data and student predictions.
- Frequency controls replace scene-specific evidence with training-marginal event/conjunction presence conditioned on the target agent, using the same effects and denominators. These are matched controls, not exhaustive calibration baselines.
- Baseline and short controls were freshly regenerated. Their entire score arrays matched the immutable pilot SHA256 hashes exactly before reusing the verified pilot AP files. Twelve new detector metric runs evaluated the four full-data conditions across3seeds.
- Strength1 was inherited from the frozen pilot; no new development selection or final-AP tuning occurred. Development proposal AP is not used as detector AP.

## Mining and pair semantics

All annotated frames of the same64training videos were inspected: {mining['annotated_frames']:,}annotatedframes and{mining['target_actor_frames']:,}valid target actor-frames, versus{mining['short_frames']:,}annotatedframes and{mining['short_actor_frames']:,}actor-frames in the short segments. Empty-target frames do not create transactions. All original short-segment weights and marginals were reproduced exactly (maximum absolute error{mining['short_max_abs_reproduction_error']}). This supplies additional ground-truth supervision; it does not expand the video identities or retrain the student.

Single rules use identical video/target-agent balancing, support≥5actor-frames in≥3videos,5video-equivalent shrinkage, parent/child exclusions and bounded log-odds effects. Full mining yields{mining['single_rules']:,}single rules. The frozen pair procedure considered{mining['pair_candidates']:,}nonhierarchical event pairs, found{mining['pair_candidate_rules']:,}effect/support-qualified pair→target rules and retained{mining['pair_rules']:,}rules across{mining['pair_distinct_antecedents']:,}distinct antecedent pairs. At most16rules per consequent are retained by training-only video support, effect magnitude and class-ID tie-breaks. pair-candidates.csv includes discarded candidates as well as retained ones; single-rules.csv lists all full single rules.

A pair means two different observed event labels among eligible other actors; both may occur on the same neighboring actor. Identity and hierarchical duplex/triplet implications are excluded between antecedents and consequent. Pair effects are nonzero only when the shrunk pair conditional is above both single conditionals or below both. The residual log-odds effect measures the increment beyond the closer extreme single conditional, then clips to[-1,1]. This is one specified practical rule design, not a universal test of every association-mining method.

At inference, context is drawn once from original current-frame scores, confidence≥0.25, class-agnostic NMS0.5, at most30neighbors, normalized center distance≤0.5, IoU<0.3 and intersection/minarea<0.5. Per-event evidence is the maximum compatible neighbor score. Pair evidence is min(evidenceA,evidenceB); this agrees with Boolean conjunction but is a soft inference heuristic. The weighted pair contribution is added to the full single contribution and clipped[-1,1]. No corrected scores feed back into context. Primitive outputs remain bit-identical, final detector confidence is applied once, and GRU state resets only at video boundaries.

Rule coverage:

'''
 for name,cov in coverage.items():report+=f"- {name}: {cov['duplex']}/49duplex, {cov['triplet']}/86triplet, {cov['tail47']}/47tail, {cov['deep28']}/28deep and{cov['common39']}/39common consequents.\n"
 report+='\n## Interpretation\n\n'
 for name,meaning in [('full_single-minus-short_single','Expanded mining'),('full_pair-minus-full_single','Adding two-event rules'),('full_pair-minus-baseline','The full combined correction')]:
  d=contrasts[name]['triplet'];direction='increased' if d['mean']>0 else 'decreased' if d['mean']<0 else 'did not change';report+=f"{meaning} {direction} mean triplet mAP by{d['mean']:+.3f}percentage points, with paired seed differences "+', '.join(f'{x:+.3f}' for x in d['deltas'])+'. '
 report+='\n\n'
 for name in protocol['primary_contrasts']:
  k='-minus-'.join(name);c=changes['tail47'][k];report+=f"For {k}, tail-class mean AP improves for{c['improve']}classes, declines for{c['decline']}and ties for{c['tie']}. "
 d=contrasts['full_pair-minus-full_pair_frequency']['triplet'];report+=f"\n\nThe combined contextual correction differs from its context-free frequency control by{d['mean']:+.3f}triplet points. A gain relative to baseline can mix contextual information and calibration; the frequency comparison is necessary to assess this distinction. No claim is made that this identifies the GRU's internal representations.\n"
 report+='''
## Limitations and reproducibility

The198-video official benchmark has prior research use and historical detector checkpoint-selection exposure. Student/semantic-teacher training/selection are disjoint from these videos, but this is not an untouched full-system test. Student development videos overlap historical teacher development. No new independent-test claim is made.

Ground-truth mining and uncertain predicted context differ. Overlapping actor-frame transactions are dependent; video support is descriptive, not a statistical independence guarantee. Spatial proximity does not imply interaction, causation, intention or future events. Three reused checkpoints are paired evaluation replicates, not three new training runs. No deployment-safety claim or teacher-free retraining claim is made. No unplanned architecture, fine-tuning or loss experiment was added.

Protocol: protocol.json. Frozen source/rule/input/checkpoint hashes: run-root freeze.json. Environment and original source revision: provenance.json. Exact split manifest, asset-audit.json, preflight.json, pilot-replay.json, replay-audit.json and final-audit.json preserve validation. No thesis, defense slides, original checkpoints or working inference bundle were changed. There are no new training curves because this is a fixed-model postprocessing study.
'''
 if runtime:
  rr=runtime['summary'];report+=f'''\n## Raw-video runtime\n\nThe full-pair adapter processed{runtime['frames']}frames across{runtime['videos']}complete videos,10Hz sampling, from raw MP4 with no teacher/features/future frames. The packaged model tensor hash matches the evaluated seed0checkpoint, and an independent NumPy replay checked all output scores. Memory bounds and video-boundary resets passed.\n\nFull-pair correction median{rr['correction']['median_ms']:.3f}ms, p95{rr['correction']['p95_ms']:.3f}ms. Serial total median{rr['total']['median_ms']:.3f}ms, p95{rr['total']['p95_ms']:.3f}ms; file-processing throughput{runtime['total_fps']:.2f}frames/s. The total includes decoding, production detector, student, correction and compressed JSON export, excludes camera/display/transport, and is not a claim of online end-to-end latency. Per-component timing, model loading, sample sizes/resolution/candidate counts and GPU memory are in runtime-summary.json and per-video report.json. Detector AP uses the fixed full-candidate manifest; production runtime uses confidence0.25/NMS0.7/max300.\n'''
 (out/'FINDINGS.md').write_text(report)
 fig,axes=plt.subplots(1,3,figsize=(15,4));x=np.arange(len(CONDITIONS));axes[0].errorbar(x,[summary[c]['triplet']['mean'] for c in CONDITIONS],yerr=[summary[c]['triplet']['sample_sd'] for c in CONDITIONS],fmt='o',capsize=3);axes[0].set_xticks(x,CONDITIONS,rotation=60,ha='right');axes[0].set_ylabel('Triplet detector mAP (%) ± sample SD')
 triplets=[r for r in rows if r['group']=='triplet']
 for ax,k,title in [(axes[1],'full_single-minus-short_single','More mining frames'),(axes[2],'full_pair-minus-full_single','Two-event increment')]:ax.bar(range(86),[r[k] for r in triplets],color=['#28788c' if r['tail47'] else '#777777' for r in triplets]);ax.axhline(0,color='black',lw=.6);ax.set(title=title,xlabel='All86triplet class IDs',ylabel='Mean class AP change (pp)')
 fig.tight_layout();fig.savefig(out/'comparisons.png',dpi=180);fig.savefig(out/'comparisons.pdf');plt.close(fig)
 if runtime:
  bench=read(ROOT/'preflight.json')['benchmark'];fig,ax=plt.subplots(figsize=(6,4))
  for c,x in [('baseline',0),('full_single',bench['full_single']['median_ms']),('full_pair',bench['full_pair']['median_ms'])]:ax.scatter(x,summary[c]['triplet']['mean']);ax.annotate(c,(x,summary[c]['triplet']['mean']),xytext=(4,4),textcoords='offset points')
  ax.set(xlabel='Correction+export median ms (same proposal frame)',ylabel='Triplet detector mAP (%)',title='Incremental cost; no detector/encoding in x-axis');ax.margins(.25);fig.tight_layout();fig.savefig(out/'incremental-cost.png',dpi=180);plt.close(fig)
 progress('analysis-complete',contrasts={k:v['triplet']['mean'] for k,v in contrasts.items()})
if __name__=='__main__':main()
