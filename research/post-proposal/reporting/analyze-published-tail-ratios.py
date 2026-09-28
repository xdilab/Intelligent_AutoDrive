"""Recover an existing RetinaNet tail baseline and audit paired expert/blend gains.
No inference, training, checkpoint selection, or validation-based weight selection.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json, re, statistics as st

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wiki',type=Path,default=Path('/data/repos/wiki'));ap.add_argument('--output',type=Path);a=ap.parse_args();w=a.wiki;o=a.output or w/'artifacts/published-tail-comparison-20260928';o.mkdir(exist_ok=True,parents=True)
 baseline=Path('/data/repos/ROAD_Reason/experiments/exp11_yolo/results_i3d_fullcand_baseline.json');counts=w/'artifacts/proposal-tail/train-counts.json';manifest=w/'artifacts/contextual-roi-20260917/data/shared-frames.json';selection=w/'artifacts/contextual-evolution-20260924/simplified/selection.json'
 sources={str(p):sha(p) for p in [baseline,counts,manifest,selection]};m=read(manifest);labels=m['labels']['triplet'];z={r['label']:r['z'] for r in read(counts)['rows']};assert len(labels)==86 and set(labels)==set(z)
 groups={'tail47':[i for i,n in enumerate(labels) if z[n]<0],'deep28':[i for i,n in enumerate(labels) if z[n]<-.5],'common39':[i for i,n in enumerate(labels) if z[n]>=0]};assert [len(v) for v in groups.values()]==[47,28,39]
 for f in [w/'artifacts/post-proposal-experiments/train-counts.json',w/'artifacts/contextual-roi-20260917/data/train-counts.json']:
  zz={r['label']:r['z'] for r in read(f)['rows']};assert zz==z;sources[str(f)]=sha(f)
 def metric(v):
  assert len(v)==86 and all(0<=x<=100 for x in v)
  d={'triplet':st.mean(v),**{g:st.mean(v[i] for i in ix) for g,ix in groups.items()}}
  d['tail_triplet_ratio']=d['tail47']/d['triplet'];d['deep_triplet_ratio']=d['deep28']/d['triplet'];d['tail_common_ratio']=d['tail47']/d['common39'];return d
 def summarize(id,name,runs):
  keys=list(runs[0]['metrics']);means={k:st.mean(r['metrics'][k] for r in runs) for k in keys};sd={k:st.stdev(r['metrics'][k] for r in runs) if len(runs)>1 else None for k in keys}
  return {'id':id,'name':name,'n':len(runs),'mean':means,'sample_sd':sd,'ratio_of_mean_tail_to_mean_triplet':means['tail47']/means['triplet'],'runs':runs}
 b=read(baseline);assert b['n_frames']==36717 and b['rows']=='i3d_own_fullcand_top300';bp={line.split(' : ')[0]:float(line.split(' : ')[-1]) for line in b['per_class']['triplet']};assert set(bp)==set(labels);bm=metric([bp[n] for n in labels]);assert abs(bm['triplet']-b['summary']['triplet'])<1e-6
 models={'retinanet':summarize('retinanet','3D-RetinaNet/I3D, local replication',[{'seed':None,'metrics':bm}])};sel=read(selection)
 # Include all13 available contextual conditions; headline comparisons below are prespecified parent/blend pairs.
 for id,meta in sel['models'].items():
  source=Path(meta['source']);sources[str(source)]=sha(source);assert sources[str(source)]==meta['source_sha256'];raw=read(source);rows=[r for r in raw['runs'] if r.get('kind')==meta['condition'] or r.get('run','').rsplit('-seed',1)[0]==meta['condition']];assert len(rows)==3
  runs=[]
  for r in rows:
   assert r['n_frames']==36717 and r['frame_sha256']==m['frame_sha256'] and r['candidate_sha256']==m['candidate_sha256']
   v=r['per_class_AP']['triplet'];d=metric(v);assert abs(d['triplet']-r['summary']['triplet'])<1e-6
   for g in groups:assert abs(d[g]-r['tail'][g]['mAP'])<1e-6
   seed=r.get('seed');seed=int(re.search(r'-seed(\d+)$',r['run']).group(1)) if seed is None else seed
   runs.append({'seed':seed,'metrics':d,'signature':r['signature']})
  assert {r['seed'] for r in runs}=={0,1,2};model=summarize(id,meta['condition'],sorted(runs,key=lambda r:r['seed']))
  for k in ['triplet',*groups]:assert abs(model['mean'][k]-meta['mean'][k])<1e-6
  models[id]=model
 pairs=[]
 for blend,expert in [('94','89'),('94','93'),('104','102'),('104','103')]:
  bs={r['seed']:r['metrics'] for r in models[blend]['runs']};es={r['seed']:r['metrics'] for r in models[expert]['runs']};deltas=[{'seed':s,**{k:bs[s][k]-es[s][k] for k in bs[s]}} for s in sorted(bs)]
  pairs.append({'blend':blend,'expert':expert,'mean_delta':{k:st.mean(d[k] for d in deltas) for k in bs[0]},'wins_out_of_3':{k:sum(d[k]>0 for d in deltas) for k in bs[0]},'seed_deltas':deltas})
 report={'protocol':'Existing detector AP@IoU0.5; identical36717 frames; native RetinaNet candidates vs fixedYOLO candidates for contextual experts/blends. No new inference.','group_definition':'Training-frequency z of log10 counts over86 triplets; tail47 z<0, deep28 z<-.5, common39 z>=0.','ratio_definition':'Descriptive Tail47 AP / overall86 Triplet AP. Main ratio is ratio of means; mean and SD of seed ratios also retained. Not share of detections or AP contribution. Overall includes tail classes.','sources_sha256':sources,'frame_audit':read(o/'frame-audit.json'),'evaluator_audit':read(o/'evaluator-audit.json'),'memberships':{g:[labels[i] for i in ix] for g,ix in groups.items()},'models':models,'paired_comparisons':pairs,'limits':['RetinaNet is one local trained replication, not an author checkpoint; contextual means use3seeds.','RetinaNet-vs-contextual comparison does not isolate blending, because architecture, detector and training differ.','Blend-vs-parent pairs isolate the effect of the development-selected probability blend for these fixed checkpoints.','Ratio can increase when overall AP decreases. Use absolute overall/tail/deep AP and paired deltas as primary evidence.','No significance claim;3training seeds do not measure dataset sampling uncertainty.','DCB blend improves over each DCB expert but has lower absolute tail/deep AP than focal blend.','FRCB has no verified matched tail result here.']}
 (o/'analysis.json').write_text(json.dumps(report,indent=2))
 columns=['id','name','n','triplet','tail47','deep28','common39','tail_triplet_ratio_of_means','mean_seed_tail_triplet_ratio','sd_seed_tail_triplet_ratio']
 with (o/'comparison.csv').open('w') as f:
  wr=csv.writer(f);wr.writerow(columns)
  for id,x in models.items():wr.writerow([id,x['name'],x['n'],*[x['mean'][k] for k in ['triplet','tail47','deep28','common39']],x['ratio_of_mean_tail_to_mean_triplet'],x['mean']['tail_triplet_ratio'],x['sample_sd']['tail_triplet_ratio']])
 lines=['# Matched tail aggregation and blend comparison','',report['protocol'],'','Tail/Triplet is descriptive; it is not evidence of superiority by itself.','', '| Model | n | Triplet AP | Tail47 AP | Deep28 AP | Tail/Triplet |','|---|---:|---:|---:|---:|---:|']
 for id in ['retinanet','89','93','94','102','103','104']:
  x=models[id];d=x['mean'];lines.append(f"| {id}: {x['name']} | {x['n']} | {d['triplet']:.3f} | {d['tail47']:.3f} | {d['deep28']:.3f} | {100*x['ratio_of_mean_tail_to_mean_triplet']:.2f}% |")
 lines+=['','## Paired evidence','', '| Blend − parent | Δ triplet pp | Δ tail pp | Δ deep pp | Tail wins /3 |','|---|---:|---:|---:|---:|']
 for x in pairs:
  d=x['mean_delta'];lines.append(f"| {x['blend']} − {x['expert']} | {d['triplet']:+.3f} | {d['tail47']:+.3f} | {d['deep28']:+.3f} | {x['wins_out_of_3']['tail47']} |")
 lines+=['','## Interpretation','', 'The matched blend-versus-parent changes support a descriptive benefit from blending. The ratio alone does not: RetinaNet can have a higher ratio while both its overall and tail AP are lower.','',*['- '+x for x in report['limits']],'','All13 contextual conditions are included in comparison.csv; analysis.json retains per-seed ratios, paired deltas, exact class membership, signatures and source hashes.']
 (o/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:28]))
if __name__=='__main__':main()
