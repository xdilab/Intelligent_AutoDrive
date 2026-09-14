import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();root=a.root;seed=a.seed;cfg=json.loads((root/'code/protocol.json').read_text());prep=json.loads((root/'results/preparation.json').read_text());src=Path(cfg['source_study']);dest=root/f'runs/seed-{seed}';dest.mkdir(parents=True,exist_ok=True);assert torch.cuda.is_available() or cfg.get('smoke',False);device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    x=np.load(root/'data/x.npy',mmap_mode='r');y=np.load(root/'data/y.npy',mmap_mode='r');part=np.load(root/'data/partition.npy',mmap_mode='r');fold=np.load(root/'data/expert-fold.npy',mmap_mode='r');rows={n:np.flatnonzero(part==j) for j,n in enumerate(['expert','gate','development'])};E=rows['expert'];G=rows['gate'];D=rows['development'];assert len(set(E)&set(G))==0 and len(set(E)&set(D))==0
    p=torch.load(src/'data/phrase_embeds.pt',weights_only=False)['embeds'];alpha=torch.load(src/'data/flat_alphas.pt',weights_only=True).float();perm=torch.randperm(184,generator=torch.Generator().manual_seed(20260910));mat={'phrase':p,'shuffled':p[perm]};models={}
    for kind in ['flat','phrase','shuffled']:
        print('FULL HEAD',kind,flush=True);m=fit_expert(lambda:make_head(kind,mat.get(kind)),x,y,E,alpha,seed,cfg['expert_epochs'],cfg['expert_batch_size'],device);save_model(m,dest/f'head-{kind}.pt',kind=kind,seed=seed,expert_videos=prep['videos']['expert']);models[kind]=m
    # OOF inputs for composition learning are generated entirely inside expert videos.
    oof={kind:np.lib.format.open_memmap(dest/f'oof-{kind}.npy',mode='w+',dtype=np.float32,shape=(len(E),184)) for kind in ['flat','phrase']}
    for kind in ['flat','phrase']:
        for j in [0,1]:
            tr=E[fold[E]==j];local=np.flatnonzero(fold[E]==1-j);te=E[local];assert not np.intersect1d(tr,te).size
            print('OOF HEAD',kind,j,flush=True);m=fit_expert(lambda:make_head(kind,mat.get(kind)),x,y,tr,alpha,seed,cfg['expert_epochs'],cfg['expert_batch_size'],device);oof[kind][local]=predict(m,x,te,device);del m
        oof[kind].flush()
    yc=np.asarray(y[E,49:],np.float32);mlps={}
    for name in ['stage5','stage6']:
        dim=1073 if name=='stage5' else 1208;zin=np.lib.format.open_memmap(dest/f'input-{name}.npy',mode='w+',dtype=np.float32,shape=(len(E),dim));zin[:,:49]=oof['flat'][:,:49]
        if name=='stage6':zin[:,49:184]=oof['phrase'][:,49:]
        for st in range(0,len(E),16384):zin[st:st+16384,-1024:]=x[E[st:st+16384]]
        zin.flush();print('MLP',name,flush=True);m=fit_expert(lambda:make_mlp(dim),zin,yc,np.arange(len(E)),alpha[49:],seed,cfg['expert_epochs'],cfg['expert_batch_size'],device);save_model(m,dest/f'{name}.pt',in_dim=dim,seed=seed,expert_videos=prep['videos']['expert']);mlps[name]=m;del zin
    del oof,yc
    pred={}
    for split,ix in [('gate',G),('development',D)]:
        flat=predict(models['flat'],x,ix,device);phrase=predict(models['phrase'],x,ix,device);shuffled=predict(models['shuffled'],x,ix,device);z=np.concatenate([flat[:,:49],np.asarray(x[ix])],1);p5=predict(mlps['stage5'],z,np.arange(len(ix)),device);del z
        pred[split]={'stage5':p5,'phrase':phrase[:,49:],'shuffled':shuffled[:,49:],'targets':np.array(y[ix,49:],np.float32)};del flat,phrase,shuffled
        np.savez(dest/f'{split}-expert-probabilities.npz',**pred[split]);print('HELDOUT SCORED',split,len(ix),flush=True)
    result={'seed':seed,'protocol':cfg,'preparation_sha256':sha(root/'results/preparation.json'),'selection':{},'dev_reference':{}}
    yd=pred['development']['targets'];result['dev_reference']['stage5_triplet_crop_AP']=float(np.mean(crop_ap(yd[:,49:],pred['development']['stage5'][:,49:])))
    for expert in ['phrase','shuffled']:
        pg=pred['gate'];pd=pred['development'];print('GLOBAL GATE',expert,flush=True);glob=fit_gate(pg['stage5'],pg[expert],pg['targets'],seed,cfg['gate_epochs'],cfg['gate_batch_size'],device,global_gate=True);anchor=float(glob.a.item());g=glob.weights().detach().numpy();global_ap=crop_ap(yd[:,49:],((1-g)*pd['stage5']+g*pd[expert])[:,49:]);global_result={'anchor_logit':anchor,'weight':float(g[0]),'dev_triplet_crop_AP':float(np.mean(global_ap)),'dev_per_triplet_crop_AP':global_ap}
        candidates=[]
        for lam in cfg['lambdas']:
            print('CLASS GATE',expert,lam,flush=True);m=fit_gate(pg['stage5'],pg[expert],pg['targets'],seed,cfg['gate_epochs'],cfg['gate_batch_size'],device,anchor=anchor,lam=lam);g=m.weights().detach().numpy();v=crop_ap(yd[:,49:],((1-g)*pd['stage5']+g*pd[expert])[:,49:]);record={'lambda':lam,'weights':g.tolist(),'logits':m.a.detach().numpy().tolist(),'dev_triplet_crop_AP':float(np.mean(v)),'dev_per_triplet_crop_AP':v};candidates.append(record)
        selected=max(candidates,key=lambda r:(r['dev_triplet_crop_AP'],r['lambda']));out={'expert':expert,'global':global_result,'class':selected,'candidates':candidates,'no_positive_indices':np.flatnonzero(pg['targets'].sum(0)==0).tolist(),'selection_data':'development training-video partition only','seed':seed};(dest/f'gate-{expert}.json').write_text(json.dumps(out,indent=2)+'\n');result['selection'][expert]=out
    result['checkpoint_sha256']={p.name:sha(p) for p in dest.glob('*.pt')};result['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(root/f'results/train-seed{seed}.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE TRAIN',seed,flush=True)
if __name__=='__main__':main()
