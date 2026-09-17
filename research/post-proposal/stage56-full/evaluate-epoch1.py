import argparse,json,pickle,time,hashlib,multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from architecture import FullStage
from train import parts
import base,metric

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--condition',required=True);a=p.parse_args();root=Path('/work/bbyrd1/stage56-full-20260914');cfg=json.loads((root/'code/protocol.json').read_text());name=f'{a.stage}-{a.condition}';checkpoint=root/'runs'/name/'epoch-1.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['epoch']==1 and ck['position']==0 and ck['protocol']==cfg;assert ck['report']['stage']==a.stage and ck['report']['condition']==a.condition;checkpoint_hash=base.sha(checkpoint);r={'selected':'epoch-1'};src=Path(cfg['original_study']);manifest=json.loads((src/'code/shared-frames.json').read_text());keys=manifest['frames'];assert len(keys)==36717;yolo=pickle.load((src/'data/dets_v8x_best_val_fullcand.pkl').open('rb'))['records'];candidate_hash=hashlib.sha256(b''.join(s.encode()+np.asarray(yolo[s]['boxes_xyxyn']).tobytes()+np.asarray(yolo[s]['conf']).tobytes()+np.asarray(yolo[s]['cls']).tobytes() for s in keys)).hexdigest();assert candidate_hash==manifest['candidate_sha256'];ann=json.loads(Path(cfg['annotation']).read_text());torch.set_num_threads(8);m=FullStage(cfg,a.condition,a.stage)
 named=dict(m.named_parameters())
 assert set(ck['state'])=={n for n,v in m.named_parameters() if v.requires_grad}
 with torch.no_grad():
  for n,v in ck['state'].items():named[n].copy_(v)
 assert ck['data_sha256']==json.loads((root/'results/preparation.json').read_text())['data_sha256']
 print('EPOCH1_CHECKPOINT',str(checkpoint),checkpoint_hash,flush=True)
 # Save per-frame predictions with candidate-order checks for resumable final inference.
 pred=root/'predictions-epoch1'/name;pred.mkdir(parents=True,exist_ok=True)
 provenance={'protocol':cfg,'selected':r['selected'],'checkpoint_sha256':checkpoint_hash,'candidate_sha256':candidate_hash}
 marker=pred/'provenance.json'
 if marker.exists():assert json.loads(marker.read_text())==provenance
 else:marker.write_text(json.dumps(provenance,indent=2))
 with torch.no_grad():
  for i,s in enumerate(keys):
   path=pred/f'{s}.npy'
   if path.exists():continue
   v,f=s.rsplit('_',1);fid=int(f);nf=ann['db'][v]['numf'];fids=[min(max(fid-3+j,1),nf) for j in range(8)];b=yolo[s]['boxes_xyxyn'];q=yolo[s]['conf'].astype(np.float32);row={'video':v,'fid':fid,'fids':fids,'key_t':fids.index(fid),'boxes':b.tolist(),'targets':np.zeros((len(b),184),np.float32),'frame_sha256':[base.sha(base.frame_root(cfg)/v/f'{fj:05d}.jpg') for fj in fids]};scores=[]
   for part in parts(row,64):
    x,_=base.crops(part,cfg)
    with torch.autocast('cuda',dtype=torch.bfloat16):z,_,_=m(x,row['key_t'])
    scores.append(z.float().sigmoid().cpu().numpy())
   sig=np.concatenate(scores) if scores else np.empty((0,184),np.float32);sig*=q[:,None];sig[:,0]=q;temp=path.with_suffix('.tmp')
   with temp.open('wb') as file:np.save(file,sig)
   temp.replace(path)
   if i%100==0:print('INFERENCE',i,len(keys),flush=True)
 del m,ann;torch.cuda.empty_cache();payload=pickle.load((src/'data/dets_i3d_val_fullcand.pkl').open('rb'));assert payload['labels']==manifest['labels'];labels=manifest['labels'];scale=np.array([840,600,840,600],np.float32);metric.GTS=[[] for _ in metric.HEADS];metric.BOXES=[yolo[s]['boxes_xyxyn'].astype(np.float32) for s in keys];metric.AGENTS=[yolo[s]['cls'].astype(int) for s in keys];metric.SCORES=[np.load(pred/f'{s}.npy') for s in keys]
 for i,s in enumerate(keys):
  assert metric.SCORES[i].shape==(len(metric.BOXES[i]),184);v,f=s.rsplit('_',1);gt=payload['records'][v+'/'+str(int(f))]['gt'];b=gt['boxes'].astype(np.float32)/scale if gt is not None else np.empty((0,4),np.float32);metric.GTS[0].append(np.c_[b,np.zeros(len(b),np.float32)])
  for j,h in enumerate(metric.HEADS[1:],1):metric.GTS[j].append(metric.core['get_individual_labels'](b,gt[h]).astype(np.float32) if len(b) else np.empty((0,5),np.float32))
 classes=[['agentness']]+[labels[h] for h in metric.HEADS[1:]];values={h:[None]*len(cc) for h,cc in zip(metric.HEADS,classes)}
 with mp.get_context('fork').Pool(8) as pool:
  for g,c,value,line in pool.imap_unordered(metric.one_class,[(g,c,n) for g,cc in enumerate(classes) for c,n in enumerate(cc)]):values[metric.HEADS[g]][c]=value
 zs={r['label']:r['z'] for r in json.loads((src/'code/train-counts.json').read_text())['rows']};tail={}
 for group,test in [('tail47',lambda z:z<0),('deep28',lambda z:z<-.5),('common39',lambda z:z>=0)]:
  ix=[i for i,n in enumerate(labels['triplet']) if test(zs[n])];tail[group]={'classes':len(ix),'mAP':float(np.mean([values['triplet'][i] for i in ix]))}
 output={'evaluation_scope':'fixed-epoch-1-reporting-only','checkpoint_sha256':checkpoint_hash,'effective_frame_root':str(base.frame_root(cfg)),'stage':a.stage,'condition':a.condition,'selected':r['selected'],'n_frames':len(keys),'candidate_sha256':candidate_hash,'frame_sha256':manifest['frame_sha256'],'summary':{k:float(np.mean(v)) for k,v in values.items()},'tail':tail,'ap_values':values,'protocol':cfg};(root/f'results/epoch1-final-{name}.json').write_text(json.dumps(output,indent=2))
if __name__=='__main__':main()
