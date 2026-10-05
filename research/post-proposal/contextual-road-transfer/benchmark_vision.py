"""Strictly load pretrained original-size encoder plus verified ROAD final-block delta."""
from pathlib import Path
import argparse,json,time,sys
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align
from prepare_full import sha,atomic

class Encoder:
 def __init__(self,root):
  from transformers import AutoConfig
  from transformers.dynamic_module_utils import get_class_from_dynamic_module
  from huggingface_hub import hf_hub_download
  from safetensors.torch import load_file
  cfg=json.loads((root/'protocol.json').read_text());repo=cfg['encoder_model'];rev=cfg['encoder_revision']
  ac=AutoConfig.from_pretrained(repo,revision=rev,trust_remote_code=True,local_files_only=True);cls=get_class_from_dynamic_module('modeling_internvideo2encoder.InternVideo2_CLIP_small',repo,revision=rev,local_files_only=True)
  self.model=cls(ac);self.model.load_state_dict(load_file(hf_hub_download(repo,'model.safetensors',revision=rev,local_files_only=True)),strict=True)
  self.model.requires_grad_(False).eval();self.tokens=None
  prefix='encoder.vision_encoder.blocks.23.';delta_norm=0.;source=cfg['source_encoder'];changed=[]
  if source:
   assert sha(source['path'])==source['sha256'];ck=torch.load(source['path'],map_location='cpu',weights_only=False)
   assert ck['epoch']==source['selected_epoch'] and ck['report']['selected']==f"epoch-{ck['epoch']}"
   assert ck['report']['stage']=='stage6' and ck['report']['condition']==cfg['encoder_condition']
   assert ck['protocol']['revision']==rev and ck['data_sha256']==ck['report']['data_sha256']
   for split,h in source['original_training_manifest_sha256'].items():assert ck['data_sha256'][split]==h
   keys={k for k in ck['state'] if k.startswith('encoder.')};expected={prefix+k for k,_ in self.model.vision_encoder.blocks[-1].named_parameters()};assert keys==expected,(keys^expected)
   named=dict(self.model.named_parameters())
   with torch.no_grad():
    for key in sorted(keys):
     param=named[key.removeprefix('encoder.')];value=ck['state'][key];assert value.shape==param.shape and torch.isfinite(value).all()
     delta_norm+=float((param-value).double().square().sum());param.copy_(value);assert torch.equal(param,value);changed.append(key)
   assert delta_norm>0
  for block in self.model.vision_encoder.blocks:block.with_cp=False
  self.model.float().cuda().eval();self.model.vision_encoder.blocks[-1].register_forward_hook(self.capture)
  self.audit={'condition':cfg['encoder_condition'],'visual_width':1024,'blocks':24,'source_checkpoint':source,'loaded_delta_tensors':len(changed),'delta_l2':delta_norm**.5,'all_frozen':not any(p.requires_grad for p in self.model.parameters())}
  assert self.audit['all_frozen']
 def capture(self,module,inputs,output):self.tokens=output if isinstance(output,torch.Tensor) else output[0]
 @torch.inference_mode()
 def __call__(self,x,key):
  self.tokens=None
  with torch.autocast('cuda',dtype=torch.bfloat16):self.model.encode_vision(x)
  t=self.tokens;self.tokens=None;assert t is not None and t.shape[-1]==1024 and t.shape[1] in [2048,2049]
  result=t[:,-2048:].reshape(len(x),8,256,1024)[:,key].float();assert torch.isfinite(result).all();return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(8);r=a.root
 from eval_input import ClipLoader,normalize_clip,crop_batch
 receipts=[];control_feature=None
 for arm in ['frozen','classification','contrastive']:
  child=r/arm;m=Encoder(child);loader=ClipLoader({'frames':str(child/'frames')});rows=[]
  for split in ['train','val']:
   with (child/f'data/{split}.jsonl').open() as f:
    for _ in range(4):rows.append(json.loads(next(f)))
  start=time.monotonic();n=0
  for row_index,row in enumerate(rows):
   ft=normalize_clip(loader.load(row))
   for j in range(0,len(row['boxes']),64):
    x=crop_batch(ft,row['boxes'][j:j+64]);m(x,row['key_t']);n+=len(x)
   tokens=m(F.interpolate(ft,size=(224,224),mode='bilinear',align_corners=False)[None],row['key_t'])
   fmap=tokens.reshape(1,16,16,1024).permute(0,3,1,2);boxes=torch.tensor(row['boxes'],device='cuda',dtype=torch.float32).reshape(-1,4);rois=torch.cat([boxes.new_zeros(len(boxes),1),boxes*16],1)
   actor=roi_align(fmap,rois,output_size=7,aligned=True).mean((-1,-2));scene=F.adaptive_avg_pool2d(fmap,(4,4)).flatten(2).transpose(1,2)[0]
   assert actor.shape==(len(boxes),1024) and scene.shape==(16,1024) and torch.isfinite(actor).all() and torch.isfinite(scene).all()
   if row_index==0:
    feature=tokens.cpu()
    if arm=='frozen':control_feature=feature.clone()
    else:assert not torch.equal(control_feature,feature),'Adapted full-scene output did not change'
  torch.cuda.synchronize();elapsed=time.monotonic()-start
  bank=torch.load(child/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds'];assert bank.shape==(184,512)
  result={'passed':True,'time':time.time(),'audit':m.audit,'frames':len(rows),'crops':n,'seconds':elapsed,'crops_per_second_including_scene_and_io':n/elapsed,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'benchmark_scope':'eight actual train/val frames; no cache writes; extrapolation provisional'}
  atomic(result,child/'results/benchmark.json');receipts.append(result);print('TRANSFER_BENCHMARK_PASS',json.dumps(result),flush=True)
  del m;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 atomic({'passed':True,'time':time.time(),'arms':receipts},r/'results/benchmark.json')
if __name__=='__main__':
 main()
 from launch_after_benchmark import main as launch_full
 launch_full()
