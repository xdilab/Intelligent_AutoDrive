"""Resumable head-only training on immutable aligned cache arrays."""
import argparse,hashlib,json,os,random,signal,time,sys
from pathlib import Path
import numpy as np
import torch
from model import make_model,objective

OFF=[0,1,11,33,49,98,184];HEADS=['agentness','agent','action','loc','duplex','triplet']
STOP=False

def request_stop(*_):
    global STOP
    STOP=True

def atomic_torch(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def atomic_json(value,path):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

class Cache:
    def __init__(self,root,split):
        self.arrays={n:np.load(root/f'{split}-{n}.npy',mmap_mode='r') for n in ['crop','boxes','targets','frame','context','scene']}
        n=len(self.arrays['crop']);assert all(len(self.arrays[k])==n for k in ['boxes','targets','frame','context'])
    def __len__(self):return len(self.arrays['crop'])
    def batch(self,ix,device):
        a=self.arrays
        xs=[a['crop'][ix],a['context'][ix],a['scene'][a['frame'][ix]],a['boxes'][ix]]
        return [torch.from_numpy(np.array(x,copy=True)).to(device,non_blocking=True) for x in xs],torch.from_numpy(np.array(a['targets'][ix],dtype=np.float32)).to(device)

def ap_values(y,p):
    values=[]
    for c in range(184):
        yy=np.asarray(y[:,c])[np.argsort(-p[:,c],kind='stable')]>0;n=yy.sum()
        if not n:values.append(0.);continue
        precision=yy.cumsum()/np.arange(1,len(yy)+1)
        values.append(float(np.maximum.accumulate(precision[::-1])[::-1][yy].sum()/n*100))
    return values

def metrics(y,p,root):
    values=ap_values(y,p);labels=json.loads((root/'data/shared-frames.json').read_text())['labels']['triplet'];counts=json.loads((root/'data/train-counts.json').read_text());z={r['label']:r['z'] for r in counts['rows']}
    tail={}
    for name,fn in [('common39',lambda x:x>=0),('tail47',lambda x:x<0),('deep28',lambda x:x<-.5)]:
        ix=[98+i for i,label in enumerate(labels) if fn(z[label])];tail[name]={'classes':len(ix),'mAP':float(np.mean([values[i] for i in ix]))}
    return {'metric':'development crop AP (not detector AP)','all_heads':{h:float(np.mean(values[OFF[i]:OFF[i+1]])) for i,h in enumerate(HEADS)},'tail':tail,'per_class_AP':values}

@torch.no_grad()
def predict(model,data,path,batch,device,heartbeat):
    model.eval();pred=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(data),184))
    for j in range(0,len(data),batch):
        x,_=data.batch(np.arange(j,min(j+batch,len(data))),device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):out=model(*x)
        pred[j:j+len(x[0])]=out['logits'].float().sigmoid().cpu().numpy()
        if j%(batch*100)==0:heartbeat('development',j,len(data))
    pred.flush();return pred

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    root=a.root;cfg=json.loads((root/'protocol.json').read_text());marker=json.loads((root/'cache-ready.json').read_text());assert marker['passed']
    src=Path(cfg['context_source'])/'runs'/f'mlp-contrastive-all184-seed{a.seed}'/'best.pt'
    signature={'protocol_sha256':file_sha(root/'protocol.json'),'cache_marker_sha256':file_sha(root/'cache-ready.json'),'run':{'seed':a.seed,'architecture':'stage7-stage5-residual','contrastive_scope':'all184','contrastive_target':'adapted_text'},'source_sha256':{'context':file_sha(src),**{n:file_sha(root/'inputs'/f'seed-{a.seed}'/n) for n in ['head-flat.pt','stage5.pt']}},'code_sha256':{n:file_sha(Path(__file__).parent/n) for n in ['model.py','contextual.py','train_cached.py']}}
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.manual_seed_all(a.seed)
    m=make_model(root,a.seed).to(device);alpha=torch.load(root/'data/flat_alphas.pt',map_location=device,weights_only=True).float()
    params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    name=f'stage7-seed{a.seed}';dest=root/'runs'/name;dest.mkdir(parents=True,exist_ok=True);resume=dest/'resume.pt';done=dest/'complete.json'
    if done.exists():assert json.loads(done.read_text())['signature']==signature;print('ALREADY_COMPLETE',name);return
    tr=Cache(root/'data','train');dev=Cache(root/'data','dev');epoch=position=steps=0;best=-1.;report=[]
    frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    def heartbeat(phase,pos,total,**kw):atomic_json({'run':name,'phase':phase,'epoch':epoch+1,'position':pos,'total':total,'steps':steps,'time':time.time(),**kw},dest/'progress.json')
    def checkpoint(ep,result):return {'model':m.state_dict(),'seed':a.seed,'epoch':ep,'metrics':result,'signature':signature}
    def save(ep,pos):atomic_torch({'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'position':pos,'steps':steps,'best':best,'report':report,'signature':signature,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},resume)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,request_stop)
    if resume.exists():
        ck=torch.load(resume,map_location=device,weights_only=False);assert ck['signature']==signature;m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);epoch=ck['epoch'];position=ck['position'];steps=ck['steps'];best=ck['best'];report=ck['report'];torch.set_rng_state(ck['torch_rng'].cpu());random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    else:
        # Establish independent Stage5 equality on real train/dev/YOLO cache rows before training.
        parity=[]
        for split in ['train','dev','val']:
            data=Cache(root/'data',split);ix=np.unique(np.linspace(0,len(data)-1,128,dtype=int));xs,_=data.batch(ix,device)
            with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                out=m(*xs);expected=m.assemble(xs[0]);err=float((out['logits']-expected).abs().max());assert err==0
            parity.append({'split':split,'sampled_rows':len(ix),'max_logit_error':err})
        atomic_json({'passed':True,'signature':signature,'initial_parity':parity,'trainable_parameters':sum(p.numel() for p in params),'frozen_stage5_parameters':sum(p.numel() for p in m.parameters() if not p.requires_grad)},dest/'parity.json')
        m.baseline=True;scores=predict(m,dev,dest/'baseline-development-scores.npy',cfg['batch_size'],device,heartbeat);initial=metrics(dev.arrays['targets'],scores,root);m.baseline=False
        initial.update(epoch=0,condition='unchanged Stage5; zero bridge');report.append(initial);best=initial['all_heads']['triplet']
        atomic_torch(checkpoint(0,initial),dest/'best.pt')
        base=root/'runs'/f'stage5-baseline-seed{a.seed}';base.mkdir(exist_ok=True);atomic_torch(checkpoint(0,initial),base/'best.pt');atomic_json({'passed':True,'signature':signature,'baseline':True},base/'complete.json')
        save(0,0);atomic_json(report,dest/'epochs.json');print('BASELINE',name,json.dumps(initial['all_heads']),flush=True)
    for epoch in range(epoch,cfg['epochs']):
        order=np.random.default_rng(a.seed+epoch).permutation(len(tr));m.train();totals=np.zeros(3);seen=0
        while position<len(order):
            ix=order[position:position+cfg['batch_size']];x,y=tr.batch(ix,device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):o=m(*x);loss,cls,aux=objective(o,y,alpha,cfg['contrastive_weight'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();position+=len(ix);steps+=1;totals+=np.array([float(loss.detach()),float(cls.detach()),float(aux.detach())])*len(ix);seen+=len(ix)
            if steps%cfg['save_every']==0 or STOP:save(epoch,position);heartbeat('train',position,len(tr),loss=totals[0]/seen,classification=totals[1]/seen,contrastive=totals[2]/seen);print(name,epoch+1,position,len(tr),steps,flush=True)
            if STOP:sys.exit(75)
        save(epoch,position);scores=predict(m,dev,dest/'development-scores.npy',cfg['batch_size'],device,heartbeat);result=metrics(dev.arrays['targets'],scores,root);result.update(epoch=epoch+1,training_segment_mean=(totals/max(seen,1)).tolist());report.append(result)
        if result['all_heads']['triplet']>best:best=result['all_heads']['triplet'];atomic_torch(checkpoint(epoch+1,result),dest/'best.pt')
        for n,p in m.named_parameters():
            if n in frozen:assert torch.equal(p,frozen[n]),'Frozen head changed: '+n
        position=0;save(epoch+1,0);atomic_json(report,dest/'epochs.json');print('EPOCH',name,json.dumps(result['all_heads']),flush=True)
    atomic_json({'passed':True,'signature':signature,'best_triplet_crop_AP':best,'epochs':report,'frozen_heads_verified':True,'time':time.time()},done);heartbeat('complete',len(tr),len(tr))
if __name__=='__main__':main()
