"""Fixed official candidate protocol; all six heads and frequency groups."""
import argparse,importlib.util,json,multiprocessing as mp,pickle,time
from pathlib import Path
import numpy as np
import torch
from model import ContextualRoIHead
from train_cached import Cache,atomic_json,file_sha

# Reuse the exact current study evaluator, including YOLO agent-class handling.
p=Path(__file__).parent.parent/'stage56-full/metric.py'
spec=importlib.util.spec_from_file_location('context_metric',p);metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',required=True);a=ap.parse_args();root=a.root;dest=root/'runs'/a.run
    completed=json.loads((dest/'complete.json').read_text());assert completed['passed'];checkpoint=dest/'best.pt';signature={'checkpoint_sha256':file_sha(checkpoint),'cache_marker_sha256':file_sha(root/'cache-ready.json'),'metric_sha256':file_sha(p),'core_sha256':file_sha(p.parent/'baseline-core.py'),'evaluator_sha256':file_sha(Path(__file__))}
    if (dest/'detector-results.json').exists():assert json.loads((dest/'detector-results.json').read_text())['signature']==signature;return
    torch.set_num_threads(4);ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['signature']==completed['signature'];assert ck['signature']['protocol_sha256']==file_sha(root/'protocol.json');assert ck['signature']['cache_marker_sha256']==signature['cache_marker_sha256'];assert ck['signature']['code_sha256']['model.py']==file_sha(Path(__file__).parent/'model.py');bank=torch.load(root/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds'];m=ContextualRoIHead(bank,fusion=ck['fusion'],mode=ck['mode'],seed=ck['seed']).cuda().eval();m.load_state_dict(ck['model']);data=Cache(root/'data','val');n=len(data);scores_path=dest/'detector-scores.npy';progress=dest/'detector-inference.json'
    scores=np.empty((n,184),np.float32);q=np.load(root/'data/val-q.npy',mmap_mode='r');cfg=json.loads((root/'protocol.json').read_text());batch=cfg['batch_size']
    with torch.no_grad():
        for batch_index,j in enumerate(range(0,n,batch),1):
            end=min(j+batch,n);xs,_=data.batch(np.arange(j,end),'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):o=m(*xs)
            z=o['logits'].float().sigmoid().cpu().numpy();z*=q[j:end,None];z[:,0]=q[j:end];scores[j:end]=z
            if batch_index%100==0 or end==n:
                atomic_json({'phase':'detector-inference','position':end,'total':n,'time':time.time()},dest/'progress.json')
    del m;torch.cuda.empty_cache()
    official=json.loads((root/'data/shared-frames.json').read_text());gt=pickle.load((root/'data/evaluation-gt.pkl').open('rb'));rows=[json.loads(l) for l in (root/'data/val.jsonl').open()];assert [f"{r['video']}_{r['fid']:05d}" for r in rows]==official['frames']
    boxes=data.arrays['boxes'];agents=np.load(root/'data/val-agent.npy',mmap_mode='r');metric.BOXES=[];metric.AGENTS=[];metric.SCORES=[];metric.GTS=[[] for _ in metric.HEADS];scale=np.array([840,600,840,600],np.float32)
    for r,k in zip(rows,official['frames']):
        start,end=r['row_start'],r['row_end'];metric.BOXES.append(boxes[start:end]);metric.AGENTS.append(agents[start:end]);metric.SCORES.append(scores[start:end]);g=gt[k];b=g['boxes'].astype(np.float32)/scale if g is not None else np.empty((0,4),np.float32);metric.GTS[0].append(np.c_[b,np.zeros(len(b),np.float32)])
        for i,h in enumerate(metric.HEADS[1:],1):metric.GTS[i].append(metric.core['get_individual_labels'](b,g[h]).astype(np.float32) if len(b) else np.empty((0,5),np.float32))
    labels=official['labels'];classes=[['agentness']]+[labels[h] for h in metric.HEADS[1:]];values={h:[None]*len(c) for h,c in zip(metric.HEADS,classes)}
    partial=dest/'detector-metrics.json'
    if partial.exists():
        saved=json.loads(partial.read_text());assert saved['signature']==signature;values=saved['values']
    # Fork inherits read-only arrays. Wrapper is module-level to support pool pickling.
    with mp.get_context('fork').Pool(4) as pool:
        for group,cls,value,_ in pool.imap_unordered(one_class,[(g,c,name) for g,cc in enumerate(classes) for c,name in enumerate(cc) if values[metric.HEADS[g]][c] is None]):
            values[metric.HEADS[group]][cls]=value;atomic_json({'signature':signature,'values':values},partial);atomic_json({'phase':'detector-metrics','classes_done':sum(v is not None for vv in values.values() for v in vv),'time':time.time()},dest/'progress.json')
    z={r['label']:r['z'] for r in json.loads((root/'data/train-counts.json').read_text())['rows']};tail={}
    for name,fn in [('common39',lambda v:v>=0),('tail47',lambda v:v<0),('deep28',lambda v:v<-.5)]:
        ix=[i for i,label in enumerate(labels['triplet']) if fn(z[label])];tail[name]={'classes':len(ix),'mAP':float(np.mean([values['triplet'][i] for i in ix]))}
    atomic_json({'metric':'official validation detector AP@0.5','n_frames':len(rows),'signature':signature,'candidate_sha256':official['candidate_sha256'],'frame_sha256':official['frame_sha256'],'selected_epoch':ck['epoch'],'run':a.run,'summary':{h:float(np.mean(v)) for h,v in values.items()},'tail':tail,'per_class_AP':values,'time':time.time()},dest/'detector-results.json')
    atomic_json({'phase':'complete','time':time.time()},dest/'progress.json')

def one_class(task):return metric.one_class(task)
if __name__=='__main__':main()
