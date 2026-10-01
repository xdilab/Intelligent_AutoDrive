"""Matched seed, gradient, checkpoint replay and real-cache throughput checks."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
import torch
from model import ContextualRoIHead
from dcb import DCBState,objective
from train_cached import Cache,atomic_json,atomic_torch,file_sha

def check_head(bank,device):
    for seed in [0,1,2]:
        torch.manual_seed(seed);a=ContextualRoIHead(bank,fusion='attention')
        torch.manual_seed(seed);b=ContextualRoIHead(bank,fusion='attention')
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    m=a.to(device);state=DCBState().to(device)
    x=[torch.randn(4,1408,device=device),torch.randn(4,1408,device=device),torch.randn(4,16,1408,device=device),torch.tensor([[.1,.1,.5,.5]]*4,device=device)]
    y=torch.zeros(4,184,device=device);y[0,[0,1,11,33,49,98]]=1;y[1,[0,3,15,35,58,117]]=1
    o=m(*x);assert o['logits'].shape==o['contrastive_logits'].shape==(4,184)
    # Auxiliary loss alone must train both visual and adapted-text branches.
    _,_,aux=objective(o,y,torch.ones(184,device=device),.001,state,update=False);aux.backward()
    assert m.text_input_projection.weight.grad.abs().sum()>0 and m.visual_fusion[-1].weight.grad.abs().sum()>0
    assert not m.phrase_bank.requires_grad and m.phrase_bank.grad is None
    m.zero_grad(set_to_none=True);optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4)
    def step(model,opt,st):
        opt.zero_grad(set_to_none=True);o=model(*x);loss=objective(o,y,torch.ones(184,device=device),.001,st)[0]
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
    step(m,optimizer,state)
    checkpoint=copy.deepcopy({'model':m.state_dict(),'optimizer':optimizer.state_dict(),'dcb':state.state_dict()})
    restored=ContextualRoIHead(bank,fusion='attention').to(device);restored.load_state_dict(checkpoint['model'])
    opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4);opt2.load_state_dict(checkpoint['optimizer']);state2=DCBState().to(device);state2.load_state_dict(checkpoint['dcb'])
    step(m,optimizer,state);step(restored,opt2,state2)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())
    assert torch.equal(state.S,state2.S) and torch.equal(state.N,state2.N)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--synthetic',action='store_true');a=ap.parse_args();torch.set_num_threads(4)
    if a.synthetic:
        check_head(torch.randn(184,768),'cpu');print('SYNTHETIC_PREFLIGHT_PASS');return
    r=a.root;marker=json.loads((r/'cache-ready.json').read_text());assert marker['passed']
    assert marker['contract']['protocol_sha256']==file_sha(r/'protocol.json')
    for name,digest in marker['files'].items():
        assert (r/'data'/name).exists()
        # Full compaction already hashes every array; verify small identity inputs here.
        if name in ['phrase_embeds.pt','gate-partition.npy','shared-frames.json']:assert file_sha(r/'data'/name)==digest
    bank=torch.load(r/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds'].float();check_head(bank,'cuda')
    tr=Cache(r/'data','train');batch=256;timings={}
    for weight in [0,.001]:
        torch.manual_seed(0);m=ContextualRoIHead(bank,fusion='attention').cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);st=DCBState().cuda()
        start=None
        for step in range(30):
            if step==5:torch.cuda.synchronize();start=time.monotonic()
            ix=np.arange(step*batch,(step+1)*batch)%len(tr);xs,y=tr.batch(ix,'cuda');opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(m(*xs),y,torch.ones(184,device='cuda'),weight,st)[0]
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();timings[str(weight)]={'rows_per_second':25*batch/(time.monotonic()-start)}
        del m,opt,st;torch.cuda.empty_cache()
    atomic_json({'passed':True,'cache_sha256':file_sha(r/'cache-ready.json'),'time':time.time(),'matched_seed_initialization':True,'aux_gradients_both_branches':True,'checkpoint_optimizer_dcb_replay_exact':True,'benchmark':timings},r/'preflight.json')
    print('PREFLIGHT_COMPLETE',json.dumps(timings),flush=True)

if __name__=='__main__':main()
