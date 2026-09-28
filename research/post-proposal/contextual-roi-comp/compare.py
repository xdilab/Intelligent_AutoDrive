"""Paired objective-by-readout interaction; no checkpoint selection here."""
from pathlib import Path
import json,statistics,math
from train_cached import atomic_json
ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-comp-20260919')
METRICS=['triplet','tail47','deep28','common39','action','loc','duplex']
def values(d):return {m:float(d['summary'][m] if m in d['summary'] else d['tail'][m]['mAP']) for m in METRICS}
def stats(x):
 mean=statistics.mean(x);sd=statistics.stdev(x);se=sd/math.sqrt(len(x))
 return {'per_seed':x,'mean':mean,'sample_sd':sd,'t':mean/se if se else None,'ci95_unadjusted':[mean-4.30265273*se,mean+4.30265273*se],'zero_variance':sd==0}
def contrast(a,b):return {m:a[m]-b[m] for m in METRICS}
def paired(flat_class,comp_class,flat_contrast,comp_contrast):
 gc=contrast(comp_class,flat_class);gt=contrast(comp_contrast,flat_contrast)
 return {'classification_composition_gain':gc,'contrastive_composition_gain':gt,'interaction':contrast(gt,gc),'new_arm_difference':contrast(comp_contrast,comp_class)}
def main():
 cfg=json.loads((ROOT/'protocol.json').read_text());pairs=[];raw=[];hashes=set()
 for seed in cfg['seeds']:
  runs={}
  for kind,prefix in [('classification','attention-classification'),('contrastive','attention-contrastive-all184')]:
   for head,path in [('flat',Path(cfg['baseline_runs'][kind][seed])),('comp',ROOT/'runs'/f'{prefix}-comp-seed{seed}')]:
    d=json.loads((path/'detector-results.json').read_text());assert d['n_frames']==36717;hashes.add((d['frame_sha256'],d['candidate_sha256']));runs[kind+'_'+head]=values(d)
  raw.append({'seed':seed,'metrics':runs});pairs.append(paired(runs['classification_flat'],runs['classification_comp'],runs['contrastive_flat'],runs['contrastive_comp']))
 assert len(hashes)==1
 summary={kind:{m:stats([p[kind][m] for p in pairs]) for m in METRICS} for kind in pairs[0]}
 atomic_json({'passed':True,'primary':'interaction.triplet','metric':'official detector AP@0.5','seeds':cfg['seeds'],'raw':raw,'comparisons':summary,'standalone_reference_means':{'triplet_mlp_corrected':12.174049259625491,'tail47_attention_classification':6.189396082912406},'caveats':['Exploratory validation-informed follow-up; n=3','Positive interaction can occur when both composition gains are negative; inspect both gains','Readout capacity increases644096 parameters; no pure composition or primitive-mediation claim','Secondary intervals are unadjusted; nonsignificance does not establish equivalence']},ROOT/'comparison.json')
if __name__=='__main__':main()
