import argparse,json,statistics,time,csv
from pathlib import Path
from common import sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;cfg=json.loads((root/'code/protocol.json').read_text());old=Path(cfg['expert_study']);manifest=json.loads((Path(cfg['source_study'])/'code/shared-frames.json').read_text());records={};new=['ap-blend','ap-shuffled-blend'];base=['stage5','stage6','global-blend','class-gate','shuffled-global'];variants=base+new
for seed in cfg['seeds']:
 for variant in variants:
  source=root if variant in new else old;m=json.loads((source/f'results/metrics/seed{seed}-{variant}.json').read_text());assert m['n_frames']==36717 and m['frame_sha256']==manifest['frame_sha256'] and m['candidate_sha256']==manifest['candidate_sha256']
  if variant in new:assert m['protocol']==cfg and m['selection_sha256']==cfg['selection_sha256']
  else:
   training=json.loads((old/f'results/train-seed{seed}.json').read_text());assert m['protocol']==training['protocol'] and sha(old/f'results/train-seed{seed}.json')==cfg['training_report_sha256'][str(seed)]
  records[seed,variant]=m
metrics={'triplet':lambda m:m['summary']['triplet'],'duplex':lambda m:m['summary']['duplex'],'tail47':lambda m:m['tail']['tail47']['mAP'],'deep28':lambda m:m['tail']['deep28']['mAP'],'common39':lambda m:m['tail']['common39']['mAP']};rows=[]
for v in variants:
 row={'variant':v}
 for k,fn in metrics.items():
  vals=[fn(records[s,v]) for s in cfg['seeds']];row[k+'_mean']=statistics.mean(vals);row[k+'_sd']=statistics.stdev(vals)
 rows.append(row)
pairs={v:{b:{k:[fn(records[s,v])-fn(records[s,b]) for s in cfg['seeds']] for k,fn in metrics.items()} for b in ['stage5','stage6','global-blend' if v=='ap-blend' else 'shuffled-global']} for v in new}
report={'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'n_new_evaluations':6,'means':rows,'paired_differences':pairs,'protocol':cfg,'semantic_control_differences':{k:[fn(records[s,'ap-blend'])-fn(records[s,'ap-shuffled-blend']) for s in cfg['seeds']] for k,fn in metrics.items()}}
(root/'results/summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'results/summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('COMPLETE: six new evaluations verified against matched archived controls',flush=True)
