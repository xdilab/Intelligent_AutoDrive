"""All stages: one ordered manifest, one unchanged baseline AP implementation."""
import argparse,hashlib,json,multiprocessing as mp,pickle,runpy,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).parent
core=runpy.run_path(str(ROOT/'baseline-core.py'))
PhraseHead=runpy.run_path(str(ROOT/'train-repeats.py'))['PhraseHead']
HEADS=['agentness','agent','action','loc','duplex','triplet'];OFF=[0,1,11,33,49,98,184]
GTS=BOXES=SCORES=AGENTS=None;BASELINE=False

def one_class(task):
    group,c,name=task;gts=[];dets=[]
    for i,box in enumerate(BOXES):
        g=GTS[group][i];selected=g[g[:,-1]==c].copy();selected[:,-1]=0;gts.append(selected)
        if group==1 and not BASELINE:
            mask=AGENTS[i]==c;dets.append(np.concatenate([box[mask],SCORES[i][mask,0:1]],1))
        else:dets.append(np.concatenate([box,SCORES[i][:,OFF[group]+c:OFF[group]+c+1]],1))
    mean,ap,strings=core['evaluate_detections'](gts,[dets],[name],iou_thresh=.5)
    return group,c,float(ap[0]),strings[0]

def iou(a,b):
    inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
    return inter/np.clip((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-9,None)

def load_head(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('head','flat')=='flat': h=nn.Linear(ck.get('feat_dim',256),184)
    else:h=PhraseHead(ck['state']['P'],ck.get('feat_dim',1024))
    h.load_state_dict(ck['state']);return h.eval()
def load_mlp(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=nn.Sequential(nn.Linear(ck['in_dim'],512),nn.ReLU(),nn.Linear(512,135));m.load_state_dict(ck['state']);return m.eval()
def key(stem):
    v,f=stem.rsplit('_',1);return v+'/'+str(int(f))

def main():
    global GTS,BOXES,SCORES,AGENTS,BASELINE
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);ap.add_argument('--only',nargs='*');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(8)
    manifest=json.loads(a.manifest.read_text());keys=manifest['frames'];yolo=pickle.load(open(a.data/'dets_v8x_best_val_fullcand.pkl','rb'))['records'];payload=pickle.load(open(a.data/'dets_i3d_val_fullcand.pkl','rb'));baseline=payload['records'];labels=payload['labels']
    actual=hashlib.sha256(b''.join(s.encode()+np.asarray(yolo[s]['boxes_xyxyn']).tobytes()+np.asarray(yolo[s]['conf']).tobytes()+np.asarray(yolo[s]['cls']).tobytes() for s in keys)).hexdigest();assert actual==manifest['candidate_sha256'];assert labels==manifest['labels'];assert (a.smoke or len(keys)==36717) and len(set(keys))==len(keys)
    classes=[['agentness']]+[labels[h] for h in HEADS[1:]];GTS=[[] for _ in HEADS];scale=np.array([840,600,840,600],np.float32)
    for s in keys:
        gt=baseline[key(s)]['gt'];boxes=gt['boxes'].astype(np.float32)/scale if gt is not None else np.zeros((0,4),np.float32)
        GTS[0].append(np.concatenate([boxes,np.zeros((len(boxes),1),np.float32)],1))
        for j,h in enumerate(HEADS[1:],1):GTS[j].append(core['get_individual_labels'](boxes,gt[h]).astype(np.float32) if len(boxes) else np.zeros((0,5),np.float32))
    yboxes=[yolo[s]['boxes_xyxyn'].astype(np.float32) for s in keys];AGENTS=[yolo[s]['cls'].astype(int) for s in keys];configs=[(f'stage{n}',n,None,None,None) for n in range(4)]
    for seed in a.seeds:
        dest=a.runs/f'seed-{seed}'
        for kind in ['flat','phrase','shuffled']:configs.append((f'seed{seed}-head-{kind}',4,dest/f'head-{kind}.pt',None,None))
        for variant,extra in [('stage5',None),('stage6','phrase'),('fusion-flat-evidence','flat'),('fusion-shuffled','shuffled')]:configs.append((f'seed{seed}-{variant}',5,dest/'head-flat.pt',dest/f'{variant}.pt',dest/f'head-{extra}.pt' if extra else None))
    feats=None;feat_kind=None
    for name,stage,hp,mpath,extra in configs:
        if a.only and name not in a.only:continue
        output=a.out/(name+'.json')
        if output.exists():continue
        print('SCORING '+name,flush=True);start=time.time();BASELINE=stage==0;BOXES=[] if BASELINE else yboxes;SCORES=[]
        if stage in (2,3):
            hp=a.data/'head_roialign_lam0_junkneg.pt';mpath=a.data/'comp_mlp_sigfeat.pt' if stage==3 else None
        if stage>=2:
            kind='roi' if stage<4 else 'crop'
            if feat_kind!=kind:
                feats=None;feats=pickle.load(open(a.data/('roi_feats_i3d_val.pkl' if kind=='roi' else 'crop_feats_val.pkl'),'rb'))['feats'];feat_kind=kind
            head=load_head(hp);comp=load_mlp(mpath) if mpath else None;ehead=load_head(extra) if extra else None
        with torch.no_grad():
            for i,s in enumerate(keys):
                yrec=yolo[s];brec=baseline[key(s)];yb=yboxes[i];n=len(yb);q=yrec['conf'].astype(np.float32)
                if stage in (0,1):
                    ib=np.clip(brec['boxes'].astype(np.float32)/scale,0,1);bsig=1/(1+np.exp(-brec['logits'].astype(np.float32)))
                    if stage==0:
                        BOXES.append(ib);sig=bsig;sig[:,0]=brec['scores'].astype(np.float32)
                    else:
                        sig=np.zeros((n,184),np.float32)
                        if n and len(ib):
                            overlaps=iou(yb,ib);best=overlaps.argmax(1);hit=overlaps[np.arange(n),best]>=.5;sig[hit]=bsig[best[hit]]
                else:
                    fk=key(s) if stage<4 else s;dim=256 if stage<4 else 1024
                    if fk not in feats:
                        assert n==0,f'Missing nonempty frame {s}';f=np.empty((0,dim),np.float32)
                    else:f=feats[fk].astype(np.float32)
                    assert f.shape==(n,dim) and np.isfinite(f).all(),s
                    raw=head(torch.from_numpy(f)).sigmoid().numpy() if n else np.empty((0,184),np.float32);sig=raw*q[:,None]
                    if comp and n:
                        parts=[raw[:,:49]]
                        if ehead:parts.append(ehead(torch.from_numpy(f)).sigmoid().numpy()[:,49:184])
                        parts.append(f);sig[:,49:184]=comp(torch.from_numpy(np.concatenate(parts,1))).sigmoid().numpy()*q[:,None]
                if stage!=0:sig[:,0]=q
                SCORES.append(sig)
        jobs=[(g,c,n) for g,names in enumerate(classes) for c,n in enumerate(names)];values={h:[None]*len(classes[g]) for g,h in enumerate(HEADS)};strings={h:[None]*len(classes[g]) for g,h in enumerate(HEADS)}
        # Fork shares read-only feature/score arrays. Each worker runs the original AP function unchanged.
        with mp.get_context('fork').Pool(a.workers) as pool:
            for done,(g,c,val,line) in enumerate(pool.imap_unordered(one_class,jobs),1):
                values[HEADS[g]][c]=val;strings[HEADS[g]][c]=line
                if done%20==0:print(f'{name}: {done}/184 classes',flush=True)
        summary={h:float(np.mean(np.array(v,dtype=np.float32))) for h,v in values.items()}
        counts=json.loads((ROOT/'train-counts.json').read_text());zs={r['label']:r['z'] for r in counts['rows']};tail={}
        for group,test in [('tail47',lambda z:z<0),('deep28',lambda z:z<-.5),('common39',lambda z:z>=0)]:
            inds=[i for i,n in enumerate(labels['triplet']) if test(zs[n])];tail[group]={'classes':len(inds),'mAP':float(np.mean([values['triplet'][i] for i in inds]))}
        result={'name':name,'n_frames':len(keys),'frame_sha256':manifest['frame_sha256'],'candidate_sha256':actual,'summary':summary,'tail':tail,'per_class':strings,'ap_values':values,'seconds':time.time()-start,'protocol':'unchanged baseline AP at IoU .5; fixed frame order; original stage-specific candidate generation and score assembly','numpy':np.__version__,'torch':torch.__version__}
        temp=output.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output);print(json.dumps({'completed':name,'summary':summary,'tail':tail}),flush=True);SCORES=None
if __name__=='__main__':main()
