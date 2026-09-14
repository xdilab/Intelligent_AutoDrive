import json,torch
from train import load_model,loss_fn
m=load_model();m.eval();x=torch.randn(2,8,3,224,224,device='cuda');texts=['the car turns left','the car stops at a red light']
with torch.autocast('cuda',dtype=torch.bfloat16):
 v=m.encode_vision(x);t=m.encode_text(m.tokenizer(texts).cuda());loss=loss_fn(v,t,texts,m.temp)
loss.backward()
g={prefix:sum(float(p.grad.float().square().sum()) for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5 for prefix in ['vision_encoder.blocks.','text_encoder.transformer.']}
assert all(val>0 for val in g.values()),g
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
print(json.dumps({'loss':float(loss),'video_shape':list(v.shape),'text_shape':list(t.shape),'grad_norms':g,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'status':'PASS'}),flush=True)
