import argparse,csv,json,statistics,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root;cfg=json.loads((root/'code/protocol.json').read_text());manifest=json.loads((Path(cfg['source_study'])/'code/shared-frames.json').read_text());records={}
for seed in cfg['seeds']:
 for v in cfg['final_variants']:
  m=json.loads((root/f'results/metrics/seed{seed}-{v}.json').read_text());assert m['n_frames']==36717 and m['frame_sha256']==manifest['frame_sha256'] and m['candidate_sha256']==manifest['candidate_sha256'] and m['protocol']==cfg;records[seed,v]=m
metrics={'triplet':lambda m:m['summary']['triplet'],'duplex':lambda m:m['summary']['duplex'],'tail47':lambda m:m['tail']['tail47']['mAP'],'deep28':lambda m:m['tail']['deep28']['mAP'],'common39':lambda m:m['tail']['common39']['mAP']};rows=[]
for v in cfg['final_variants']:
 row={'variant':v}
 for metric,fn in metrics.items():
  vals=[fn(records[s,v]) for s in cfg['seeds']];row[metric+'_mean']=statistics.mean(vals);row[metric+'_sd']=statistics.stdev(vals)
 rows.append(row)
with (root/'results/summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
deltas={}
for v in ['stage6','global-blend','class-gate','shuffled-global','shuffled-class-gate']:
 deltas[v]={metric:[fn(records[s,v])-fn(records[s,'stage5']) for s in cfg['seeds']] for metric,fn in metrics.items()}
(root/'results/summary.json').write_text(json.dumps({'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'n_evaluations':len(records),'protocol':cfg,'means':rows,'paired_deltas_vs_stage5':deltas,'interpretation_limits':['matched 70-percent-training-video experts, not full-data historical models','one fixed video partition; head seed variation only','internal selection uses cached-crop AP, not final detector AP','official validation previously informed the hypothesis; no claim of a fresh blind test']},indent=2)+'\n');print('COMPLETE',len(records),'evaluations verified',flush=True)
