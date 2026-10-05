"""Freeze source identities; audit video partitions before any feature extraction."""
from pathlib import Path
import argparse,hashlib,json,time

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(d,p):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def link(src,dest):
 if not dest.exists():dest.symlink_to(src)
 assert dest.resolve()==src.resolve(),dest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
 source=Path('/work/bbyrd1/contextual-roi-1b-20260930');old=Path('/work/bbyrd1/stage56-full-20260914');original=Path('/work/bbyrd1/contextual-roi-20260917')
 original_prep=json.loads((original/'preparation.json').read_text());assert sha(original/'data/phrase_embeds.pt')==original_prep['data_sha256']['phrase_embeds.pt']
 cfg=json.loads((r/'code/protocol.json').read_text());prep=json.loads((source/'preparation.json').read_text());assert prep['passed'];oldprep=json.loads((old/'results/preparation.json').read_text());sets={};oldhash={}
 for split in ['train','dev']:
  p=old/f'data/{split}.jsonl';oldhash[split]=sha(p);assert oldhash[split]==oldprep['data_sha256'][split]
  assert {json.loads(l)['video'] for l in p.open()}==set(prep['splits'][split]['videos'])
 for split,n in [('train',420),('dev',90),('gate',90),('val',198)]:
  sets[split]=set(prep['splits'][split]['videos']);assert len(sets[split])==n,(split,len(sets[split]))
 for x in sets:
  for y in sets:
   if x!=y:assert not sets[x]&sets[y]
 assert prep['frame_sha256']=='733ce2ce4bbbb59966fe3d588c9749e4e123290c7083d4af9aa1273f32a758b9'
 for name,h in prep['source_sha256'].items():assert sha(source/'data'/name)==h,name
 ckmeta={}
 for arm in ['classification','contrastive']:
  checkpoint=old/f'runs/stage6-{arm}/best.pt';report=json.loads((old/f'results/stage6-{arm}.json').read_text())
  assert report['passed'] and report['selected']=='epoch-3' and report['protocol']['revision']==cfg['encoder_revision']
  assert report['data_sha256']==oldprep['data_sha256']
  ckmeta[arm]={'path':str(checkpoint),'sha256':sha(checkpoint),'selected_epoch':3,'source_report_sha256':sha(old/f'results/stage6-{arm}.json'),'original_training_manifest_sha256':oldhash,'upstream_loss':'focal + triplet86/fixed-text auxiliary' if arm=='contrastive' else 'focal only','encoder_seed':0}
 for arm in ['frozen','classification','contrastive']:
  child=r/arm
  for name in ['data','results','runs','inputs']: (child/name).mkdir(parents=True,exist_ok=True)
  link(r/'code',child/'code');link(source/'frames',child/'frames');link(source/'inputs/gate',child/'inputs/gate')
  for name,h in prep['source_sha256'].items():link(source/'data'/name,child/'data'/name)
  # Only original-size phrase bank: never the 1B phrase bank.
  link(original/'data/phrase_embeds.pt',child/'data/phrase_embeds.pt')
  protocol=dict(cfg,root=str(child),encoder_condition=arm,source_encoder=ckmeta.get(arm),phrase_bank_sha256=sha(child/'data/phrase_embeds.pt'))
  pp=child/'protocol.json'
  if pp.exists():assert json.loads(pp.read_text())==protocol
  else:atomic(protocol,pp)
  receipt={**prep,'source_preparation_sha256':sha(source/'preparation.json'),'encoder_source':ckmeta.get(arm),'source_checkpoint_selection':'original90 development videos; reused for new heads; gate and val never in encoder training','time':time.time()}
  if not (child/'preparation.json').exists():atomic(receipt,child/'preparation.json')
 audit={'passed':True,'time':time.time(),'video_counts':{k:len(v) for k,v in sets.items()},'disjoint_video_sets':True,'source_encoders':ckmeta,'upstream_encoder_seeds':1,'head_seeds':[0,1,2],'source_data_hashes':prep['source_sha256'],'prepared_roots':[str(r/a) for a in ['frozen','classification','contrastive']]}
 atomic(audit,r/'results/input-audit.json');print('TRANSFER_INPUT_AUDIT_PASS',json.dumps(audit),flush=True)
if __name__=='__main__':main()
