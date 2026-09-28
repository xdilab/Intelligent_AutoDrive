from pathlib import Path
import json,statistics,time
from train_cached import atomic_json,file_sha
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-concat-20260924')
def stats(v):
 mean=statistics.mean(v);sd=statistics.stdev(v) if len(v)>1 else None
 ci=[mean-4.302652729911275*sd/len(v)**.5,mean+4.302652729911275*sd/len(v)**.5] if len(v)==3 else None
 return {'mean':mean,'sample_sd':sd,'paired95CI':ci,'values':v}
def flat(d):return dict(d['summary'],**{k:v['mAP'] for k,v in d['tail'].items()})
def main():
 cfg=json.loads((ROOT/'protocol.json').read_text());rows=[]
 for kind in ['classification','contrastive-all184','global']:
  for seed in cfg['seeds']:
   name=f'attention-{kind}-dcb-seed{seed}';paths=[ROOT/'runs'/name/'detector-results.json',Path(cfg['baseline_root'])/'runs'/name/'detector-results.json']
   if not paths[0].exists():continue
   new,old=[json.loads(p.read_text()) for p in paths]
   for key in ['n_frames','candidate_sha256','frame_sha256']:assert new[key]==old[key]
   a,b=flat(new),flat(old);rows.append({'kind':kind,'seed':seed,'new':a,'baseline':b,'delta':{k:a[k]-b[k] for k in a},'sources':[{'path':str(p),'sha256':file_sha(p)} for p in paths]})
 summary={}
 for kind in ['classification','contrastive-all184','global']:
  rs=[r for r in rows if r['kind']==kind]
  if rs:summary[kind]={'n':len(rs),'paired_delta':{k:stats([r['delta'][k] for r in rs]) for k in rs[0]['delta']}}
 interaction={}
 pairs=[(next((r for r in rows if r['kind']=='classification' and r['seed']==seed),None),next((r for r in rows if r['kind']=='contrastive-all184' and r['seed']==seed),None)) for seed in cfg['seeds']]
 pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
 if pairs:interaction={k:stats([b['delta'][k]-a['delta'][k] for a,b in pairs]) for k in pairs[0][0]['delta']}
 atomic_json({'interaction':interaction,'passed':len(rows)==9,'completed_comparisons':len(rows),'raw':rows,'summary':summary,'metric':'official detector AP@0.5; paired seeds','caveat':'New projection adds1,310,720 trainable weights; scope combines connectivity and capacity.','time':time.time()},ROOT/'comparison.json')
if __name__=='__main__':main()
