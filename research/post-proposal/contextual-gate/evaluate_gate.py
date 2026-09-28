"""Official fixed candidates. Reconstruct gated scores in RAM; retain inputs/checkpoints."""
import argparse,importlib.util,json,multiprocessing as mp,pickle,time
import numpy as np
import torch
from common import ROOT,BASE,parents,signature,atomic_json,file_sha
from router import Router,composition_map,actor_features
p=__import__('pathlib').Path(__file__).parent.parent/'stage56-full/metric.py';s=importlib.util.spec_from_file_location('gate_metric',p);metric=importlib.util.module_from_spec(s);s.loader.exec_module(metric)
def one_class(task):return metric.one_class(task)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--kind',required=True,choices=['global','class','generic','factorized']);a=ap.parse_args();torch.set_num_threads(4);dest=ROOT/'runs'/f'seed{a.seed}';sig=signature(a.seed);kind=a.kind
 ckpath=dest/'global.json' if kind=='global' else dest/kind/'best.pt';evalsig={'study':sig,'checkpoint':file_sha(ckpath),'evaluator':file_sha(__file__),'metric':file_sha(p),'core':file_sha(p.parent/'baseline-core.py')};output=dest/f'{kind}-detector-results.json'
 if output.exists():assert json.loads(output.read_text())['signature']==evalsig;return
 paths=parents(a.seed);old=[json.loads((q/'detector-results.json').read_text()) for q in paths];assert old[0]['frame_sha256']==old[1]['frame_sha256'] and old[0]['candidate_sha256']==old[1]['candidate_sha256']

 for path,result,digest in zip(paths,old,sig['parents']):assert result['signature']['checkpoint_sha256']==digest
 aa,bb=[np.load(q/'detector-scores.npy',mmap_mode='r') for q in paths];q=np.load(BASE/'data/val-q.npy',mmap_mode='r');n=len(q);assert aa.shape==bb.shape==(n,184);scores=np.empty((n,184),np.float32);official=json.loads((BASE/'data/shared-frames.json').read_text());mapping=composition_map(official['labels']);m=None
 for result in old:assert (result['frame_sha256'],result['candidate_sha256'])==(official['frame_sha256'],official['candidate_sha256'])
 if kind=='global':ck=json.loads(ckpath.read_text());assert ck['signature']==sig;weight=ck['weight']
 else:
  ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['signature']==sig;m=Router(mapping,torch.tensor(ck['support']),ck['anchor'],kind).cuda().eval();m.load_state_dict(ck['model']);m.requires_grad_(False)
  if kind=='class':weight=(m.anchor+m.shrink*m.offset).sigmoid().detach().cpu().numpy()
 crop=np.load(BASE/'data/val-crop.npy',mmap_mode='r');context=np.load(BASE/'data/val-context.npy',mmap_mode='r');boxes=np.load(BASE/'data/val-boxes.npy',mmap_mode='r');batch=256 if kind in ['generic','factorized'] else 16384
 with torch.no_grad():
  for start in range(0,n,batch):
   end=min(start+batch,n);scores[start:end,:49]=aa[start:end,:49]
   if kind in ['generic','factorized']:
    denom=np.maximum(q[start:end,None],1e-12);pa=np.array(aa[start:end]/denom);pb=np.array(bb[start:end]/denom);actor=actor_features(torch.from_numpy(np.array(crop[start:end])).cuda(),torch.from_numpy(np.array(context[start:end])).cuda(),None)
    classes=torch.arange(135,device='cuda').repeat(end-start);inputs=[torch.from_numpy(x).cuda().repeat_interleave(135,0) for x in [pa,pb]];weight=m(*inputs,actor.repeat_interleave(135,0),classes)[1].reshape(end-start,135).cpu().numpy()
   scores[start:end,49:]=(1-weight)*aa[start:end,49:]+weight*bb[start:end,49:]
   if (start//batch)%100==0 or end==n:atomic_json({'phase':'detector-inference','kind':kind,'position':end,'total':n,'time':time.time()},dest/'progress.json')
 if m is not None:del m
 torch.cuda.empty_cache();gt=pickle.load((BASE/'data/evaluation-gt.pkl').open('rb'));rows=[json.loads(l) for l in (BASE/'data/val.jsonl').open()];assert len(rows)==36717 and [f"{r['video']}_{r['fid']:05d}" for r in rows]==official['frames'];agents=np.load(BASE/'data/val-agent.npy',mmap_mode='r');metric.BOXES=[];metric.AGENTS=[];metric.SCORES=[];metric.GTS=[[] for _ in metric.HEADS];scale=np.array([840,600,840,600],np.float32)
 for row,key in zip(rows,official['frames']):
  start,end=row['row_start'],row['row_end'];metric.BOXES.append(boxes[start:end]);metric.AGENTS.append(agents[start:end]);metric.SCORES.append(scores[start:end]);g=gt[key];b=g['boxes'].astype(np.float32)/scale if g is not None else np.empty((0,4),np.float32);metric.GTS[0].append(np.c_[b,np.zeros(len(b),np.float32)])
  for i,h in enumerate(metric.HEADS[1:],1):metric.GTS[i].append(metric.core['get_individual_labels'](b,g[h]).astype(np.float32) if len(b) else np.empty((0,5),np.float32))
 labels=official['labels'];classes=[['agentness']]+[labels[h] for h in metric.HEADS[1:]];values={h:[None]*len(c) for h,c in zip(metric.HEADS,classes)};partial=dest/f'{kind}-metric-progress.json'
 if partial.exists():
  oldp=json.loads(partial.read_text());assert oldp['signature']==evalsig;values=oldp['values']
 with mp.get_context('fork').Pool(4) as pool:
  for group,cls,value,_ in pool.imap_unordered(one_class,[(g,c,name) for g,cc in enumerate(classes) for c,name in enumerate(cc) if values[metric.HEADS[g]][c] is None]):
   values[metric.HEADS[group]][cls]=value;atomic_json({'signature':evalsig,'values':values},partial);atomic_json({'phase':'detector-metrics','kind':kind,'classes_done':sum(v is not None for vs in values.values() for v in vs),'time':time.time()},dest/'progress.json')
 z={r['label']:r['z'] for r in json.loads((BASE/'data/train-counts.json').read_text())['rows']};tail={}
 for name,fn in [('common39',lambda v:v>=0),('tail47',lambda v:v<0),('deep28',lambda v:v<-.5)]:
  ix=[i for i,label in enumerate(labels['triplet']) if fn(z[label])];tail[name]={'classes':len(ix),'mAP':float(np.mean([values['triplet'][i] for i in ix]))}
 atomic_json({'passed':True,'signature':evalsig,'metric':'official detector AP@0.5','n_frames':len(rows),'frame_sha256':official['frame_sha256'],'candidate_sha256':official['candidate_sha256'],'summary':{h:float(np.mean(v)) for h,v in values.items()},'tail':tail,'per_class_AP':values,'time':time.time(),'kind':kind,'seed':a.seed},output)
 print('EVALUATED',a.seed,kind,flush=True)
if __name__=='__main__':main()
