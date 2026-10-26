"""Fixed ten-seed report, with original and additional cohorts disclosed separately."""
from context_common import *
import csv,statistics,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METRICS=['triplet','duplex','tail47','deep28','common39']
COHORTS={'original_0_2':list(range(3)),'added_3_9':list(range(3,10)),'combined_0_9':list(range(10))}
T95={3:4.302652729911275,7:2.4469118511449692,10:2.2621571627409915}
def value(r,k):return r['summary'][k] if k in ['triplet','duplex'] else r['tail'][k]['mAP']
def stats(xs):return dict(values=xs,mean=statistics.mean(xs),sample_sd=statistics.stdev(xs))
def paired(a,b):
 d=[x-y for x,y in zip(a,b)];m=statistics.mean(d);sd=statistics.stdev(d);half=T95[len(d)]*sd/math.sqrt(len(d));return dict(deltas=d,mean=m,sample_sd=sd,ci95=[m-half,m+half],df=len(d)-1,interval='Paired t; fixed reused benchmark; exploratory result-informed extension, multiple contrasts unadjusted')
def main():
 verify();out=ROOT/'analysis';out.mkdir(exist_ok=True);protocol=read(HERE/'protocol.json');pairs=protocol['primary_contrasts']+protocol['control_contrasts'];records={(c,s):read(ROOT/'runs'/f'{c}-seed{s}'/'detector-results.json') for s in SEEDS for c in CONDITIONS};assert all(r['signature']==dict(signature(),checkpoint_sha256=sha(checkpoint(s))) for (c,s),r in records.items());cohorts={}
 for name,seeds in COHORTS.items():
  conditions={c:{k:stats([value(records[c,s],k) for s in seeds]) for k in METRICS} for c in CONDITIONS};contrasts={a+'-minus-'+b:{k:paired(conditions[a][k]['values'],conditions[b][k]['values']) for k in METRICS} for a,b in pairs};cohorts[name]=dict(seeds=seeds,conditions=conditions,contrasts=contrasts)
 combined=cohorts['combined_0_9'];labels=read(BASE_ROOT/'data-manifest.json')['labels'];subsets=groups(labels['triplet']);rows=[]
 for g in ['duplex','triplet']:
  for i,label in enumerate(labels[g]):
   row=dict(group=g,class_id=i,label=label,tail47=g=='triplet' and i in subsets['tail47'],deep28=g=='triplet' and i in subsets['deep28'],gt_instances=records['baseline',0]['gt_counts'][g][i])
   for c in CONDITIONS:
    vals=[records[c,s]['per_class_AP'][g][i] for s in SEEDS];row[c+'_mean']=statistics.mean(vals)
    for s,v in zip(SEEDS,vals):row[f'{c}_seed{s}']=v
   for a,b in pairs:row[a+'-minus-'+b]=row[a+'_mean']-row[b+'_mean']
   rows.append(row)
 with (out/'per-class.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (out/'per-seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['seed','condition']+METRICS);w.writeheader();w.writerows([dict(seed=s,condition=c,**{k:value(records[c,s],k) for k in METRICS}) for s in SEEDS for c in CONDITIONS])
 changes={}
 for name,subset in [('triplet',[r for r in rows if r['group']=='triplet']),('tail47',[r for r in rows if r['tail47']]),('deep28',[r for r in rows if r['deep28']]),('common39',[r for r in rows if r['group']=='triplet' and not r['tail47']]),('duplex',[r for r in rows if r['group']=='duplex'])]:changes[name]={key:dict(improve=sum(r[key]>1e-8 for r in subset),decline=sum(r[key]<-1e-8 for r in subset),tie=sum(abs(r[key])<=1e-8 for r in subset)) for key in combined['contrasts']}
 runtime=read(ROOT/'runtime-summary.json');source=read(CONTEXT_ROOT/'analysis/comparison.json');result=dict(time=time.time(),signature=signature(),metric='Official detector AP@IoU0.5 (%)',seeds=SEEDS,cohorts=cohorts,conditions=combined['conditions'],contrasts=combined['contrasts'],changed_classes=changes,rule_coverage=source['rule_coverage'],runtime_reused_from_parent=runtime,frames=36717,videos=198,candidates=6163968,processed_frames=38663,triplets_supported=85,extension_requested_after_original_outcomes=True);save(out/'comparison.json',result)
 table=['| Condition | Triplet | Duplex | Tail47 | Deep28 | Common39 |','|---|---:|---:|---:|---:|---:|']
 for c in CONDITIONS:table.append('| '+c+' | '+' | '.join(f"{combined['conditions'][c][k]['mean']:.3f} ± {combined['conditions'][c][k]['sample_sd']:.3f}" for k in METRICS)+' |')
 ct=['| Contrast | Original 3: mean [95% interval] | Added 7: mean [95% interval] | All 10: mean [95% interval] |','|---|---:|---:|---:|']
 for key in combined['contrasts']:
  values=[]
  for name in COHORTS:
   d=cohorts[name]['contrasts'][key]['triplet'];values.append(f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
  ct.append('| '+key+' | '+' | '.join(values)+' |')
 report='''# Full-video mining and two-event rules: ten-seed study

Completed COMP 765 extension, started October 26, 2026. All ten original distilled student checkpoints (seeds 0–9) were evaluated under seven fixed conditions. No student retraining, architecture change, rule retuning or checkpoint promotion occurred. The original three-seed block remains immutable.

The user requested ten seeds after the first three outcomes were visible. The seven additional seeds and unchanged protocol were fixed before their corrected-condition evaluations, and all were completed regardless of outcomes. Combined intervals are exploratory; this is not a retrospective confirmatory preregistration or a new independent video test.

## Detector results

Official detector AP at IoU 0.5, percent, mean ± sample standard deviation across ten paired checkpoints:

'''+ '\n'.join(table)+ '\n\nTriplet changes in percentage points, with nominal paired-t intervals (df 2, 6 and 9 respectively):\n\n'+ '\n'.join(ct)+'\n\n'
 for key,label in [('full_single-minus-short_single','Additional mining frames'),('full_pair-minus-full_single','The incremental two-event rules'),('full_pair-minus-baseline','The complete contextual correction')]:
  d=combined['contrasts'][key]['triplet'];tail=combined['contrasts'][key]['tail47'];counts=changes['tail47'][key];report+=f"{label}: mean triplet change **{d['mean']:+.3f} points**, 95% paired interval [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]. Tail47 changes by {tail['mean']:+.3f} points; {counts['improve']} tail classes improve, {counts['decline']} decline and {counts['tie']} tie in mean AP.\n\n"
 extension=combined['conditions']['full_pair']['triplet']['mean']-combined['conditions']['short_single']['triplet']['mean'];report+=f"The combined full-video/pair condition differs from the original short-rule method by {extension:+.3f} triplet points. This descriptive difference is not a substitute for the separately declared data-scale and rule-complexity contrasts above.\n\n"
 for key,label in [('full_single-minus-full_frequency','Single-event context versus its frequency control'),('full_pair-minus-full_pair_frequency','Two-event context versus its frequency control')]:
  d=combined['contrasts'][key]['triplet'];report+=f"{label}: {d['mean']:+.3f} triplet points, interval [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]. "
 report+='''

These controls use training-marginal evidence with the same fixed rule effects. A baseline gain can combine scene information with calibration; the controls help assess that distinction without proving a unique mechanism. No non-significant interval is interpreted as equivalence. The intervals describe variability across these training seeds conditional on one reused benchmark, not uncertainty over new videos. Multiple endpoints/contrasts are exploratory and unadjusted.

## Fixed study

- Architecture: frozen ImageNet ResNet50 through layer3, shared 448×672 visual features, actor RoI/geometry/agent features and a 256-dimensional scene GRU. Exactly 9,651,864 semantic parameters; detector parameters counted separately. The teacher is absent at inference, and only present/past frames update state.
- Same seven conditions: baseline, short_single, short_frequency, full_single, full_frequency, full_pair, full_pair_frequency. All use the same checkpoint within each seed. Strength remains 1 from the original pilot; no new AP tuning.
- Full mining uses all 12,270 annotated frames in the same 64 training videos (11,948 nonempty target frames; 269,600 target actor-frames), versus 2,036 annotated frames and 45,766 actor-frames in the original segments. This is extra ground-truth supervision, not new video identities or student retraining.
- Frozen rules: 3,970 full single-event rules plus 1,532 residual pair rules over 395 distinct antecedent pairs. Two event labels may occur on the same or different eligible neighboring actors. Identity/parent-child implications are excluded. Minimum support is 5 actor-frames and 3 videos, with video/agent balancing, shrinkage and a fixed cap of 16 pair rules per consequent. All retained and discarded qualified pair candidates remain available in pair-candidates.csv.
- Pair effects add only when their shrunk target conditional lies above both single conditionals or below both. Inference uses min(original event evidences), one pass, no feedback, bounded combined log-odds change, and unchanged primitive outputs. Final detector confidence is applied once.
- Same 198 official validation videos, 36,717 scored frames, 38,663 causal prefix steps and 6,163,968 fixed candidates. 85/86 triplets have evaluation GT; all 86 remain in the mean. Tail47/Deep28/Common39 use the authoritative training-frequency manifest. No mis-indexed child arrays are used.
- Original 21 result files were reused after their complete score/integrity audit. Seven additional baseline arrays were regenerated and matched their original whole-file SHA256 hashes before reusing baseline AP. All 42 additional corrected-condition detector evaluations were computed. Complete per-seed and all 135 event-class outcomes are in per-seed.csv and per-class.csv.

## Runtime and reproducibility

The audited, prespecified seed-0 raw-video adapter is unchanged. Its prior 593-frame, three-video runtime/replay record is reused by hash; no new speed improvement is claimed from adding seeds. It accepts raw MP4 without the teacher, cached features or future frames. Candidates and primitive outputs matched the verified baseline exactly, and an independent NumPy full-pair replay agreed within the recorded 1e-6 tolerance. Video resets, finiteness and retained-memory checks passed.

'''
 rt=runtime['summary'];report+=f"Measured full-pair correction median {rt['correction']['median_ms']:.3f} ms, p95 {rt['correction']['p95_ms']:.3f} ms. Serial raw-video total median {rt['total']['median_ms']:.3f} ms, p95 {rt['total']['p95_ms']:.3f} ms, including decoding, detector, student, correction and compressed JSON export. This file-processing measurement excludes camera/display/transport and is not a claim of end-to-end online latency. Production candidate filtering differs from the fixed full-candidate detector AP protocol.\n\n"
 report+='''Exact commands and paths are in README.md. protocol.json defines the extension; freeze.json pins code, rules, splits, original result sources and all ten checkpoints. preflight.json/provenance.json record input equivalence, environment and source revisions. Each batch preserves score hashes, baseline replay, all per-class results, image identities and verified cleanup receipts. The final audit independently checks all aggregate arithmetic, per-class rows, frozen assets and checkpoint backups.

## Limits

The official validation set has prior research use and detector checkpoint-selection exposure. Student/semantic-teacher training/selection are disjoint from it, but it is not an untouched full-system test. More seeds improve measurement of training randomness; they do not remove dataset reuse or establish generalization to new video populations. All seeds share one fixed mined rule bank; these intervals do not measure uncertainty from resampling mining data or constructing different rules.

Dependent actor-frame transactions and uncertain predicted context differ from ground-truth mining. Nearby actors need not interact. Associations do not establish causes, intentions, future events or the GRU's internal representations. The pair rule design and cap are one fixed practical implementation, not an exhaustive search. No deployment-safety claim is made. No thesis, slides, original checkpoints, production bundle or remote branch were modified.
'''
 (out/'FINDINGS.md').write_text(report)
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 for c in ['baseline','short_single','full_single','full_pair']:axes[0].plot(SEEDS,combined['conditions'][c]['triplet']['values'],marker='o',label=c)
 axes[0].set(xlabel='Seed',ylabel='Triplet detector mAP (%)',xticks=SEEDS);axes[0].legend(fontsize=8)
 tr=[r for r in rows if r['group']=='triplet']
 for ax,key,title in [(axes[1],'full_single-minus-short_single','More mining frames'),(axes[2],'full_pair-minus-full_single','Two-event increment')]:ax.bar(range(86),[r[key] for r in tr],color=['#28788c' if r['tail47'] else '#777777' for r in tr]);ax.axhline(0,color='black',lw=.7);ax.set(title=title,xlabel='All 86 triplet class IDs',ylabel='Mean class AP change (pp)')
 from matplotlib.patches import Patch
 axes[1].legend(handles=[Patch(color='#28788c',label='Tail47'),Patch(color='#777777',label='Common39')],fontsize=8)
 fig.tight_layout();fig.savefig(out/'comparisons.png',dpi=180);fig.savefig(out/'comparisons.pdf');plt.close(fig)
 bench=read(CONTEXT_ROOT/'preflight.json')['benchmark'];fig,ax=plt.subplots(figsize=(6,4))
 for c,x in [('baseline',0),('full_single',bench['full_single']['median_ms']),('full_pair',bench['full_pair']['median_ms'])]:
  ax.scatter(x,combined['conditions'][c]['triplet']['mean']);ax.annotate(c,(x,combined['conditions'][c]['triplet']['mean']),xytext=(4,4),textcoords='offset points')
 ax.set(xlabel='Same-frame correction+export median ms (reused benchmark)',ylabel='Ten-seed mean triplet detector mAP (%)',title='Incremental rule cost; detector/encoding excluded');ax.margins(.25);fig.tight_layout();fig.savefig(out/'incremental-cost.png',dpi=180);plt.close(fig)
 progress('analysis-complete',seeds=10,contrasts={k:v['triplet']['mean'] for k,v in combined['contrasts'].items()})
if __name__=='__main__':main()
