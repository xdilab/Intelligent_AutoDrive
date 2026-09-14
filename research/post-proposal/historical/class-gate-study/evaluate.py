"""Final untouched-manifest evaluation of matched experts and preselected gates."""
import argparse,hashlib,json,multiprocessing as mp,pickle,runpy,time
from pathlib import Path
import numpy as np
import torch
from common import *
HEADS=['agentness','agent','action','loc','duplex','triplet'];OFF=[0,1,11,33,49,98,184]
GTS=BOXES=SCORES=AGENTS=CORE=None

def one_class(task):
    group,c,name=task;gts=[];dets=[]
    for i,b in enumerate(BOXES):
        g=GTS[group][i];g=g[g[:,-1]==c].copy();g[:,-1]=0;gts.append(g)
        if group==1:
            mask=AGENTS[i]==c;dets.append(np.concatenate([b[mask],SCORES[i][mask,0:1]],1))
        else:dets.append(np.concatenate([b,SCORES[i][:,OFF[group]+c:OFF[group]+c+1]],1))
    _,ap,strings=CORE['evaluate_detections'](gts,[dets],[name],iou_thresh=.5)
    return group,c,float(ap[0]),strings[0]

def main():
    global GTS,BOXES,SCORES,AGENTS,CORE
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();root=a.root;cfg=json.loads((root/'code/protocol.json').read_text());src=Path(cfg['source_study']);dest=root/f'runs/seed-{a.seed}';out=root/'results/metrics';out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(a.workers);CORE=runpy.run_path(str(src/'code/baseline-core.py'));manifest=json.loads((src/'code/shared-frames.json').read_text());keys=manifest['frames'];labels=manifest['labels'];assert len(keys)==36717 or cfg.get('smoke',False)
    training=json.loads((root/f'results/train-seed{a.seed}.json').read_text());assert training['protocol']==cfg
    gates={k:json.loads((dest/f'gate-{k}.json').read_text()) for k in ['phrase','shuffled']}
    for k,g in gates.items():
        best=max(g['candidates'],key=lambda r:(r['dev_triplet_crop_AP'],r['lambda']));assert g['class']==best
        w=np.array(g['class']['weights']);assert w.shape==(135,) and np.isfinite(w).all() and ((w>=0)&(w<=1)).all()
        assert all(abs(w[i]-g['global']['weight'])<1e-6 for i in g['no_positive_indices'])
    yolo=pickle.load(open(src/'data/dets_v8x_best_val_fullcand.pkl','rb'))['records'];candidate_sha=hashlib.sha256(b''.join(s.encode()+np.asarray(yolo[s]['boxes_xyxyn']).tobytes()+np.asarray(yolo[s]['conf']).tobytes()+np.asarray(yolo[s]['cls']).tobytes() for s in keys)).hexdigest();assert candidate_sha==manifest['candidate_sha256'];payload=pickle.load(open(src/'data/dets_i3d_val_fullcand.pkl','rb'));assert payload['labels']==labels;baseline=payload['records'];classes=[['agentness']]+[labels[h] for h in HEADS[1:]];GTS=[[] for _ in HEADS];scale=np.array([840,600,840,600],np.float32);BOXES=[yolo[s]['boxes_xyxyn'].astype(np.float32) for s in keys];AGENTS=[yolo[s]['cls'].astype(int) for s in keys];qs=[yolo[s]['conf'].astype(np.float32) for s in keys];del yolo
    for s in keys:
        v,f=s.rsplit('_',1);gt=baseline[v+'/'+str(int(f))]['gt'];b=gt['boxes'].astype(np.float32)/scale if gt is not None else np.zeros((0,4),np.float32);GTS[0].append(np.c_[b,np.zeros(len(b),np.float32)])
        for j,h in enumerate(HEADS[1:],1):GTS[j].append(CORE['get_individual_labels'](b,gt[h]).astype(np.float32) if len(b) else np.zeros((0,5),np.float32))
    del baseline,payload
    feats=pickle.load(open(src/'data/crop_feats_val.pkl','rb'))['feats'];models={k:load_head(dest/f'head-{k}.pt') for k in ['flat','phrase','shuffled']};models.update({k:load_mlp(dest/f'{k}.pt') for k in ['stage5','stage6']});counts=json.loads((src/'code/train-counts.json').read_text());zs={r['label']:r['z'] for r in counts['rows']};flat_result=None
    for variant in VARIANTS:
        output=out/f'seed{a.seed}-{variant}.json'
        if output.exists():
            if variant=='head-flat':flat_result=json.loads(output.read_text())
            continue
        print('SCORING',variant,flush=True);start=time.time();SCORES=[]
        with torch.no_grad():
            for st in range(0,len(keys),64):
                en=min(st+64,len(keys));parts=[];ns=[]
                for i in range(st,en):
                    n=len(BOXES[i]);f=feats.get(keys[i],np.empty((0,1024),np.float32));assert f.shape==(n,1024) and np.isfinite(f).all();parts.append(f);ns.append(n)
                x=torch.from_numpy(np.concatenate(parts).astype(np.float32));q=np.concatenate(qs[st:en]);raw=models['phrase' if variant=='head-phrase' else 'flat'](x).sigmoid();sig=raw.numpy()*q[:,None]
                if variant not in ['head-flat','head-phrase']:
                    if variant=='stage6':
                        phrase=models['phrase'](x).sigmoid()[:,49:];p=models['stage6'](torch.cat([raw[:,:49],phrase,x],1)).sigmoid().numpy()
                    else:
                        p=models['stage5'](torch.cat([raw[:,:49],x],1)).sigmoid().numpy()
                        if variant!='stage5':
                            expert='shuffled' if variant.startswith('shuffled') else 'phrase';language=models[expert](x).sigmoid().numpy()[:,49:];gate=gates[expert];w=np.float32(gate['global']['weight']) if variant in ['global-blend','shuffled-global'] else np.array(gate['class']['weights'],np.float32);p=(1-w)*p+w*language
                    sig[:,49:]=p*q[:,None]
                sig[:,0]=q;offset=0
                for n in ns:SCORES.append(sig[offset:offset+n]);offset+=n
        assert len(SCORES)==len(keys)
        reuse=variant not in ['head-flat','head-phrase'];values={h:list(flat_result['ap_values'][h]) if reuse and gi<4 else [None]*len(classes[gi]) for gi,h in enumerate(HEADS)};strings={h:list(flat_result['per_class'][h]) if reuse and gi<4 else [None]*len(classes[gi]) for gi,h in enumerate(HEADS)};tasks=[(g,c,n) for g,cc in enumerate(classes) if not reuse or g>=4 for c,n in enumerate(cc)]
        with mp.get_context('fork').Pool(a.workers) as pool:
            for done,(g,c,val,line) in enumerate(pool.imap_unordered(one_class,tasks),1):
                values[HEADS[g]][c]=val;strings[HEADS[g]][c]=line
                if done%30==0:print(variant,done,'/',len(tasks),'classes',flush=True)
        summary={h:float(np.mean(np.array(v,np.float32))) for h,v in values.items()};tail={}
        for group,test in [('tail47',lambda z:z<0),('deep28',lambda z:z<-.5),('common39',lambda z:z>=0)]:
            ix=[i for i,n in enumerate(labels['triplet']) if test(zs[n])];tail[group]={'classes':len(ix),'mAP':float(np.mean([values['triplet'][i] for i in ix]))}
        result={'name':f'seed{a.seed}-{variant}','seed':a.seed,'variant':variant,'n_frames':len(keys),'frame_sha256':manifest['frame_sha256'],'candidate_sha256':candidate_sha,'summary':summary,'tail':tail,'ap_values':values,'per_class':strings,'seconds':time.time()-start,'protocol':cfg,'selection_sha256':{k:sha(dest/f'gate-{k}.json') for k in gates},'expert_partition':'70 percent of training videos; matched expert-only controls','numpy':np.__version__,'torch':torch.__version__};temp=output.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output)
        if variant=='head-flat':flat_result=result
        SCORES=None;print('COMPLETE EVAL',variant,json.dumps(summary),flush=True)
if __name__=='__main__':main()
