"""Matched seeds and objective-only intervention, exact optimizer/RNG resume."""
from common import *
def sequence(model,seq):
 state=None;zs=[]
 for r in seq:
  z,state=model(r['roi'],r['scene'],r['boxes'],r['agents'],state);zs.append(z)
 return torch.cat(zs),torch.cat([r['labels'] for r in seq])
def update(model,opt,seq,kd):
 opt.zero_grad(set_to_none=True);z,y=sequence(model,seq)
 if not len(z):return None
 t=torch.cat([r['teacher'] for r in seq]);loss=objective(z,y,t,kd)
 assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);opt.step()
 return float(loss.detach())
@torch.no_grad()
def selection(model,seqs):
 model.eval();pred=[];ys=[]
 for seq in seqs:
  z,y=sequence(model,seq);pred.append(z.sigmoid().cpu().numpy());ys.append(y.cpu().numpy())
 x,y=np.concatenate(pred),np.concatenate(ys)
 return float(-(y[:,1:]*np.log(x[:,1:].clip(1e-7,1-1e-7))+(1-y[:,1:])*np.log((1-x[:,1:]).clip(1e-7,1))).mean())
def capture(model,opt,**meta):
 return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),**meta)
def restore(model,opt,ck):
 model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
def load_data():
 m=read(ROOT/'data-manifest.json');assert read(ROOT/'preparation-complete.json')['data_manifest_sha256']==sha(ROOT/'data-manifest.json');out={'train':[],'selection':[]}
 for r in m['sequences']:
  assert sha(r['path'])==r['sha256'];seq=torch.load(r['path'],map_location='cpu',weights_only=True)
  assert len(seq)==32;assert all(('teacher' in row)==(r['split']=='train') for row in seq)
  out[r['split']].append([to_gpu(row) for row in seq])
 return out
def main():
 setup();verify();data=load_data();cfg=read(HERE/'protocol.json');overall=time.perf_counter()
 for seed in cfg['seeds']:
  rng=np.random.RandomState(seed);orders=[rng.permutation(64).tolist() for _ in range(cfg['epochs'])];order_hash=hashlib.sha256(json.dumps(orders).encode()).hexdigest();save(ROOT/f'order-seed{seed}.json',orders)
  paired=[]
  for condition,kd in [('supervised',0.),('distilled',1.)]:
   run=f'{condition}-seed{seed}';dest=ROOT/'runs'/run;dest.mkdir(parents=True,exist_ok=True);sig=dict(signature(),data_manifest_sha256=sha(ROOT/'data-manifest.json'),seed=seed,condition=condition,order_sha256=order_hash)
   if (dest/'training-complete.json').exists():
    done=read(dest/'training-complete.json');assert done['signature']==sig and done['checkpoint_sha256']==sha(dest/'best.pt');paired.append(done['initialization_sha256']);continue
   setup(seed);model=Student().cuda();initial=tensor_hash(model.state_dict());paired.append(initial);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
   epoch=0;offset=0;best=float('inf');history=[];losses=[];started=time.perf_counter()
   if (dest/'last.pt').exists():
    ck=torch.load(dest/'last.pt',map_location='cpu',weights_only=False);assert ck['signature']==sig and ck['initialization_sha256']==initial
    restore(model,opt,ck);epoch=ck['epoch'];offset=ck['offset'];best=ck['best'];history=ck['history'];losses=ck['losses'];progress('training-resumed',run=run,epoch=epoch,offset=offset)
   def state():return capture(model,opt,signature=sig,initialization_sha256=initial,epoch=epoch,offset=offset,best=best,history=history,losses=losses)
   while epoch<cfg['epochs']:
    model.train()
    for idx in range(offset,64):
     loss=update(model,opt,data['train'][orders[epoch][idx]],kd)
     if loss is not None:losses.append(loss)
     offset=idx+1
     if offset%16==0:
      p.atomic_save(state(),dest/'last.pt');progress('training',run=run,epoch=epoch+1,sequence=offset,total_sequences=64,loss=loss,elapsed=time.perf_counter()-started)
    bce=selection(model,data['selection']);epoch+=1;offset=0
    history.append(dict(epoch=epoch,loss=float(np.mean(losses)),selection_bce=bce));losses=[]
    if bce<best:best=bce;p.atomic_save(state(),dest/'best.pt')
    p.atomic_save(state(),dest/'last.pt');save(dest/'history.json',history);progress('epoch-complete',run=run,**history[-1])
   ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
   save(dest/'training-complete.json',dict(signature=sig,initialization_sha256=initial,order_sha256=order_hash,checkpoint_sha256=sha(dest/'best.pt'),selected_epoch=ck['epoch'],best_selection_bce=ck['best'],time=time.time(),seconds=time.perf_counter()-started,trainable_parameters=sum(v.numel() for v in model.parameters())))
   del model,opt;torch.cuda.empty_cache()
  assert paired[0]==paired[1],f'Initialization mismatch seed{seed}'
 save(ROOT/'training-complete.json',dict(time=time.time(),seconds=time.perf_counter()-overall,runs=2*len(cfg['seeds']),paired_initializations=True,paired_orders=True));progress('all-trained',seconds=time.perf_counter()-overall)
if __name__=='__main__':main()
