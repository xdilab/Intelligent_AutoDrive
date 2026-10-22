"""Shared full detector adapter, exact existing score assembly and AP core."""
import argparse, importlib.util, multiprocessing as mp, pickle, time
import numpy as np
import torch
from common import *

spec=importlib.util.spec_from_file_location('comp765_metric',PARENT/'stage56-full/metric.py')
metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
def one_class(task):
    start=time.monotonic();answer=metric.one_class(task)
    return (*answer,time.monotonic()-start)

def load_validation(root):
    start=time.monotonic();official=read(INPUTS['frames']);keys=official['frames'];labels=official['labels']
    d=pickle.load(INPUTS['val'].open('rb'));yr=pickle.load(INPUTS['yolo'].open('rb'))['records'];payload=pickle.load(INPUTS['gt'].open('rb'))
    assert payload['labels']==labels
    metric.BOXES=[];metric.AGENTS=[];metric.GTS=[[] for _ in HEADS];features=[];conf=[];scale=np.array([840,600,840,600],np.float32)
    candidate=hashlib.sha256()
    for stem in keys:
        r=yr[stem];v,f=stem.rsplit('_',1);g=payload['records'][v+'/'+str(int(f))]['gt'];box=r['boxes_xyxyn'].astype(np.float32)
        for b in [stem.encode(),np.asarray(r['boxes_xyxyn']).tobytes(),np.asarray(r['conf']).tobytes(),np.asarray(r['cls']).tobytes()]:candidate.update(b)
        feat=d['feats'].get(stem,np.empty((0,1024),np.float16));assert feat.shape==(len(box),1024) and np.isfinite(feat).all()
        features.append(feat);conf.append(r['conf'].astype(np.float32));metric.BOXES.append(box);metric.AGENTS.append(r['cls'].astype(int))
        gb=g['boxes'].astype(np.float32)/scale if g is not None else np.empty((0,4),np.float32)
        metric.GTS[0].append(np.c_[gb,np.zeros(len(gb),np.float32)])
        for i,h in enumerate(HEADS[1:],1):metric.GTS[i].append(metric.core['get_individual_labels'](gb,g[h]).astype(np.float32) if len(gb) else np.empty((0,5),np.float32))
    assert candidate.hexdigest()==official['candidate_sha256']
    x=np.concatenate(features);q=np.concatenate(conf);offsets=np.cumsum([0]+[len(v) for v in features]);del d,payload,yr,features,conf
    progress(root,'evaluation-loaded',frames=len(keys),rows=len(x),seconds=time.monotonic()-start)
    return x,q,offsets,official

def assemble(raw,q):
    scores=raw*q[:,None];scores[:,0]=q
    return scores

def infer(root,model,x,q,offsets,run):
    scores=np.empty((len(x),184),np.float32);start=time.monotonic()
    with torch.no_grad():
        for i in range(0,len(x),16384):
            stop=min(i+16384,len(x));raw=model(torch.from_numpy(x[i:stop].astype(np.float32)).cuda()).sigmoid().cpu().numpy()
            scores[i:stop]=assemble(raw,q[i:stop])
            if i%(16384*50)==0 or stop==len(x):progress(root,'detector-inference',run=run,position=stop,total=len(x))
    assert np.isfinite(scores).all()
    metric.SCORES=[scores[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    return scores,time.monotonic()-start

def run_evaluation(root,condition,seed,x,q,offsets,official,workers):
    dest=root/'runs'/f'{condition}-seed{seed}';ckpt=dest/'final.pt';done=read(dest/'training-complete.json')
    sig=dict(signature(root),checkpoint_sha256=sha(ckpt));assert sig['checkpoint_sha256']==done['checkpoint_sha256']
    result=dest/'detector-results.json'
    if result.exists():assert read(result)['signature']==sig;return
    ck=torch.load(ckpt,map_location='cpu',weights_only=False);model=PhraseHead(ck['state']['P']).cuda().eval();model.load_state_dict(ck['state'])
    scores,inference_seconds=infer(root,model,x,q,offsets,dest.name);del model;torch.cuda.empty_cache()
    labels=official['labels'];classes=[['agentness']]+[labels[h] for h in HEADS[1:]];values={h:[None]*len(c) for h,c in zip(HEADS,classes)};details={h:[None]*len(c) for h,c in zip(HEADS,classes)};timings=[]
    partial=dest/'detector-partial.json'
    if partial.exists():
        prior=read(partial);assert prior['signature']==sig;values=prior['values'];details=prior['details'];timings=prior['timings']
    tasks=[(g,c,name) for g,cc in enumerate(classes) for c,name in enumerate(cc) if values[HEADS[g]][c] is None]
    start=time.monotonic()
    with mp.get_context('fork').Pool(workers) as pool:
        for g,c,ap,string,seconds in pool.imap_unordered(one_class,tasks):
            values[HEADS[g]][c]=ap;details[HEADS[g]][c]=string;timings.append(dict(group=HEADS[g],class_id=c,seconds=seconds))
            save(partial,dict(signature=sig,values=values,details=details,timings=timings))
            progress(root,'detector-metrics',run=dest.name,classes_done=sum(v is not None for vv in values.values() for v in vv),total=184)
    triplet=values['triplet'];tail={name:dict(classes=len(ix),mAP=float(np.mean([triplet[i] for i in ix]))) for name,ix in groups(labels['triplet']).items()}
    save(result,dict(metric='detector frame AP@0.5 (%)',signature=sig,run=dest.name,seed=seed,condition=condition,selected_epoch=10,n_frames=len(official['frames']),candidate_sha256=official['candidate_sha256'],frame_sha256=official['frame_sha256'],summary={h:float(np.mean(v)) for h,v in values.items()},tail=tail,per_class_AP=values,per_class_details=details,inference_seconds=inference_seconds,metric_seconds=time.monotonic()-start,time=time.time()))
    metric.SCORES=None;del scores;progress(root,'detector-complete',run=dest.name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();setup();verify_freeze(a.root)
    x,q,offsets,official=load_validation(a.root)
    for seed in [0,1,2]:
        for condition in ['single','multi']:run_evaluation(a.root,condition,seed,x,q,offsets,official,a.workers)

if __name__=='__main__':main()
