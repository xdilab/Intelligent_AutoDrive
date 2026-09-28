from pathlib import Path
import json,statistics,time
from train_cached import atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-dcb-20260921');W=ROOT.parent

def stats(v):
 mean=statistics.mean(v);sd=statistics.stdev(v);h=4.302652729911275*sd/(len(v)**.5)
 return {'mean':mean,'sample_sd':sd,'paired95CI':[mean-h,mean+h],'values':v}
def flat(x):return dict(x['summary'],**{k:v['mAP'] for k,v in x['tail'].items()})
def main():
 raw=[];results=[]
 gate=json.loads((W/'contextual-gate-20260919/all-results.json').read_text())['runs']
 for seed in range(3):
  paths={'classification_dcb':ROOT/'runs'/f'attention-classification-dcb-seed{seed}/detector-results.json','contrastive_dcb':ROOT/'runs'/f'attention-contrastive-all184-dcb-seed{seed}/detector-results.json','blend_dcb':ROOT/'runs'/f'attention-global-dcb-seed{seed}/detector-results.json','classification_focal':W/'contextual-roi-20260917/runs'/f'attention-classification-seed{seed}/detector-results.json','contrastive_focal':W/'contextual-roi-all184-20260918/runs'/f'attention-contrastive-all184-seed{seed}/detector-results.json'}
  cells={k:json.loads(p.read_text()) for k,p in paths.items()};cells['blend_focal']=next(x for x in gate if x['kind']=='global' and x['seed']==seed)
  for x in cells.values():assert x['n_frames']==36717 and x['candidate_sha256']==cells['classification_focal']['candidate_sha256'] and x['frame_sha256']==cells['classification_focal']['frame_sha256']
  results.extend(cells[k] for k in ['classification_dcb','contrastive_dcb','blend_dcb']);raw.append({'seed':seed,'metrics':{k:flat(v) for k,v in cells.items()}})
 keys=list(raw[0]['metrics']['classification_dcb']);summary={k:{m:stats([r['metrics'][k][m] for r in raw]) for m in keys} for k in raw[0]['metrics']}
 contrasts={}
 for name,a,b in [('classification_loss','classification_dcb','classification_focal'),('contrastive_loss','contrastive_dcb','contrastive_focal'),('blend','blend_dcb','blend_focal')]:contrasts[name]={m:stats([r['metrics'][a][m]-r['metrics'][b][m] for r in raw]) for m in keys}
 contrasts['interaction']={m:stats([(r['metrics']['contrastive_dcb'][m]-r['metrics']['classification_dcb'][m])-(r['metrics']['contrastive_focal'][m]-r['metrics']['classification_focal'][m]) for r in raw]) for m in keys}
 atomic_json({'passed':True,'primary':'classification_loss.tail47','raw':raw,'summary':summary,'contrasts':contrasts,'time':time.time()},ROOT/'comparison.json');atomic_json({'passed':True,'runs':results,'time':time.time()},ROOT/'all-results.json')
if __name__=='__main__':main()
