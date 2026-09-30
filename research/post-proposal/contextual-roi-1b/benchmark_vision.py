"""Strict 1B visual-weight integration and real-ROAD throughput; no training."""
from pathlib import Path
import argparse,hashlib,importlib,json,sys,time,types
import numpy as np
import torch
from torch.nn import functional as F
from transformers.dynamic_module_utils import get_class_from_dynamic_module

SMALL='OpenGVLab/InternVideo2_CLIP_S';REV='1f9fca1389fd883defc652634d95a21121c85a8c'

def sdpa(self,x):
 b,n,c=x.shape
 q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
 if self.qk_normalization:
  q=self.q_norm(q.transpose(1,2).flatten(-2)).view(b,n,self.num_heads,-1).transpose(1,2)
  k=self.k_norm(k.transpose(1,2).flatten(-2)).view(b,n,self.num_heads,-1).transpose(1,2)
 z=F.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop.p if self.training else 0.,scale=self.scale)
 return self.proj_drop(self.proj(z.transpose(1,2).reshape(b,n,c)))

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

class Encoder:
 def __init__(self,root):
  receipt=json.loads((root/'results/assets.json').read_text());assert receipt['passed']
  assets={x['kind']:x for x in receipt['files']}
  for key in ['base','clip']:
   assert sha(Path(assets[key]['path']))==assets[key]['sha256']
  cls=get_class_from_dynamic_module('internvideo2_clip_vision.InternVideo2',SMALL,revision=REV,local_files_only=True)
  module=importlib.import_module(cls.__module__);pos=importlib.import_module(cls.__module__.rsplit('.',1)[0]+'.pos_embed')
  self.model=cls(embed_dim=1408,depth=40,num_heads=16,mlp_ratio=48/11,init_values=.1,num_frames=8,use_flash_attn=False,use_fused_mlp=False,use_fused_rmsnorm=False,layerscale_no_force_fp32=True)
  base=torch.load(assets['base']['path'],map_location='cpu',weights_only=False);base=base.get('module',base.get('model',base))
  delta=torch.load(assets['clip']['path'],map_location='cpu',weights_only=False);delta=delta.get('module',delta.get('model',delta))
  # Same official Stage2 -> CLIP transfer: decoder/image-only parameters excluded.
  weights={k[len('vision_encoder.'):]:v for k,v in base.items() if k.startswith('vision_encoder.') and k[len('vision_encoder.'):] in self.model.state_dict()}
  assert 'pos_embed' in weights,'No visual base weights found'
  pos.interpolate_pos_embed(weights,self.model,orig_t_size=4,pos_name='pos_embed')
  overrides={k[len('vision_encoder.'):]:v for k,v in delta.items() if k.startswith('vision_encoder.')}
  assert overrides and all(k.startswith('clip_projector.') for k in overrides),list(overrides)
  weights.update(overrides)
  # Strict coverage: never benchmark a partially initialized encoder.
  self.model.load_state_dict(weights,strict=True);del weights,base,delta
  self.model.requires_grad_(False).eval().cuda()
  # Check attention rewrite in float32 before using bf16 fast kernels.
  attention=self.model.blocks[0].attn;x=torch.randn(2,33,1408,device='cuda')
  with torch.no_grad():
   old=attention._naive_attn(x);new=sdpa(attention,x)
  error=float((old-new).abs().max());assert torch.allclose(old,new,atol=2e-5,rtol=2e-4),error
  for block in self.model.blocks:block.attn._naive_attn=types.MethodType(sdpa,block.attn)
  self.tokens=None;self.model.blocks[-1].register_forward_hook(self.capture)
  self.audit={'attention_max_abs_error':error,'visual_parameters':sum(p.numel() for p in self.model.parameters()),'weights_strict':True,'implementation_repo':SMALL,'implementation_revision':REV,'visual_width':1408,'layers':40}
 def capture(self,module,args,out):self.tokens=out if isinstance(out,torch.Tensor) else out[0]
 @torch.inference_mode()
 def __call__(self,x,key):
  self.tokens=None
  with torch.autocast('cuda',dtype=torch.bfloat16):self.model(x.permute(0,2,1,3,4),use_image=False)
  t=self.tokens;assert t is not None and t.shape==(len(x),2049,1408),None if t is None else t.shape
  out=t[:,1:].reshape(len(x),8,256,1408)[:,key].float();self.tokens=None
  assert torch.isfinite(out).all();return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;torch.set_num_threads(8)
 sys.path.insert(0,'/work/bbyrd1/stage56-full-20260914/code')
 from eval_input import ClipLoader,normalize_clip,crop_batch
 from torchvision.ops import roi_align
 rows=[json.loads(l) for l in Path('/work/bbyrd1/stage56-full-20260914/data/dev.jsonl').open()]
 selected=[rows[i] for i in np.linspace(0,len(rows)-1,4,dtype=int)]
 val=[json.loads(l) for l in Path('/work/bbyrd1/contextual-roi-20260917/data/val.jsonl').open()]
 selected += [val[i] for i in np.linspace(0,len(val)-1,4,dtype=int)]
 cfg=json.loads(Path('/work/bbyrd1/stage56-full-20260914/code/protocol.json').read_text());loader=ClipLoader(cfg)
 model=Encoder(r);print('MODEL_READY',json.dumps(model.audit),flush=True);measure=[]
 for bs in [1,4,8,16,32,64]:
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();crop_count=0;seconds=0;total_seconds=0;scene_seconds=0
  try:
   for i,row in enumerate(selected):
    frame_start=time.monotonic();imgs=loader.load(row);ft=normalize_clip(imgs);boxes=row['boxes'][:bs]
    if not boxes:continue
    x=crop_batch(ft,boxes)
    # One warmup per batch-size, excluded from measured timings.
    if i==0:
     warm=time.monotonic();model(x,row['key_t']);torch.cuda.synchronize();frame_start+=time.monotonic()-warm
    torch.cuda.synchronize();start=time.monotonic();features=model(x,row['key_t']);torch.cuda.synchronize()
    seconds+=time.monotonic()-start;crop_count+=len(x)
    scene_start=time.monotonic()
    fmap=model(F.interpolate(ft,size=(224,224),mode='bilinear',align_corners=False)[None],row['key_t']).reshape(1,16,16,1408).permute(0,3,1,2)
    coords=torch.tensor(boxes,device='cuda',dtype=torch.float32)*16;rois=torch.cat([torch.zeros(len(coords),1,device='cuda'),coords],1)
    actor=roi_align(fmap,rois,output_size=7,aligned=True).mean((-1,-2));scene=F.adaptive_avg_pool2d(fmap,(4,4)).flatten(2).transpose(1,2)
    assert actor.shape==(len(boxes),1408) and scene.shape==(1,16,1408)
    torch.cuda.synchronize();scene_seconds+=time.monotonic()-scene_start;total_seconds+=time.monotonic()-frame_start
   info={'batch_size':bs,'crops':crop_count,'encoder_seconds':seconds,'crops_per_second':crop_count/seconds,'input_scene_and_encoder_seconds':total_seconds,'scene_seconds':scene_seconds,'sample_frames':len(selected),'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
   measure.append(info);print('BENCHMARK',json.dumps(info),flush=True)
  except torch.cuda.OutOfMemoryError:
   print('BATCH_OOM',bs,flush=True);torch.cuda.empty_cache();break
 assert measure
 out={'passed':True,'scope':'visual compatibility only; full extraction waits for text+manifest validation','time':time.time(),'gpu':torch.cuda.get_device_name(),'model':model.audit,'measurements':measure,'note':'Eight sampled real dev/val frames. Encoder-only and input-plus-scene timings both reported. Cache writes, full-frame candidate density and queue time still require allowance in full-study ETA.'}
 (r/'results/benchmark.json').write_text(json.dumps(out,indent=2));print('BENCHMARK_COMPLETE',flush=True)
if __name__=='__main__':main()
