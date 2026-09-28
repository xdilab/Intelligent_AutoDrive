"""Three paired contrasts; equivalence only for predeclared bypass endpoints."""
from pathlib import Path
import json,statistics,math
from train_cached import atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-langctl-20260919')
METRICS=['triplet','tail47','deep28','common39','action','loc','duplex']
def stats(values,margin=None):
 mean=statistics.mean(values);sd=statistics.stdev(values);half=4.30265273*sd/math.sqrt(len(values));lo,hi=mean-half,mean+half
 out={'per_seed':values,'mean':mean,'sample_sd':sd,'ci95_unadjusted':[lo,hi]}
 if margin is not None:out.update(margin=margin,equivalent=bool(lo>-margin and hi<margin),interpretation='equivalent within predeclared margin' if lo>-margin and hi<margin else 'inconclusive about equivalence; inspect interval for direction and magnitude')
 return out

def main():
 cfg=json.loads((ROOT/'protocol.json').read_text());raw=[];hashes=set()
 for seed in cfg['seeds']:
  row={}
  for name,path in [('baseline',Path(cfg['baseline_runs'][seed])),('bypass',ROOT/'runs'/f'attention-bypass-seed{seed}'),('randbank',ROOT/'runs'/f'attention-randbank-seed{seed}')]:
   d=json.loads((path/'detector-results.json').read_text());assert d['n_frames']==36717;hashes.add((d['frame_sha256'],d['candidate_sha256']));row[name]={m:float(d['summary'][m] if m in d['summary'] else d['tail'][m]['mAP']) for m in METRICS}
  raw.append({'seed':seed,'metrics':row})
 assert len(hashes)==1
 contrasts={}
 for a,b in [('bypass','baseline'),('randbank','baseline'),('randbank','bypass')]:
  name=a+'_minus_'+b;margins=cfg['equivalence_margins'].get(name,{})
  contrasts[name]={m:stats([r['metrics'][a][m]-r['metrics'][b][m] for r in raw],margins.get(m)) for m in METRICS}
 atomic_json({'passed':True,'primary':'bypass_minus_baseline.triplet','raw':raw,'contrasts':contrasts,'caveats':['n=3, one split, prior validation exposure; secondary intervals unadjusted','No equivalence margin declared for randbank contrasts','Bypass nominal capacity matched but active paths removed','Real/random comparison confounds phrase content and geometry','Explicit downstream branch only, not encoder pretraining']},ROOT/'comparison.json')
if __name__=='__main__':main()
