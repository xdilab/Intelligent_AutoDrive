"""Compatibility and exact feature extraction for the frozen 1B encoder.

The original benchmark uses native PyTorch attention/MLP/RMSNorm. Optional
FlashAttention imports are guarded (and fail loudly if ever called), avoiding
installation of an unused CUDA extension. No forward computation is replaced.
"""
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align


def optional_flash_guards():
    if importlib.util.find_spec('flash_attn') is not None:
        return
    def unavailable(*args, **kwargs):
        raise RuntimeError('FlashAttention is disabled in the verified 1B inference path')
    modules = {'flash_attn': {}, 'flash_attn.modules': {},
               'flash_attn.modules.mlp': {'FusedMLP':unavailable},
               'flash_attn.flash_attn_interface': {'flash_attn_varlen_qkvpacked_func':unavailable},
               'flash_attn.bert_padding': {'unpad_input':unavailable,'pad_input':unavailable},
               'flash_attn.ops': {}, 'flash_attn.ops.rms_norm': {'DropoutAddRMSNorm':unavailable}}
    for name, attrs in modules.items():
        module=types.ModuleType(name)
        module.__spec__=importlib.machinery.ModuleSpec(name,loader=None)
        module.__path__=[]
        for key,value in attrs.items():setattr(module,key,value)
        sys.modules[name]=module


def load_encoder(root):
    optional_flash_guards()
    path=Path(__file__).resolve().parents[1]/'contextual-roi-1b/benchmark_vision.py'
    spec=importlib.util.spec_from_file_location('cseb_verified_encoder',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.Encoder(root)


def normalize_clip(images):
    ft=torch.from_numpy(images).cuda().permute(0,3,1,2).float()/255
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    return (ft-mean)/std


def crop_batch(ft,boxes):
    # Exact stage56-full/eval_input.py expansion and RoIAlign, including minimum size.
    H,W=ft.shape[-2:];b=np.asarray(boxes);n=len(b)
    cx=(b[:,0]+b[:,2])/2*W;cy=(b[:,1]+b[:,3])/2*H
    bw=np.maximum((b[:,2]-b[:,0])*W,8)*2;bh=np.maximum((b[:,3]-b[:,1])*H,8)*2
    expanded=torch.tensor(np.stack([np.clip(cx-bw/2,0,W-1),np.clip(cy-bh/2,0,H-1),
                                   np.clip(cx+bw/2,1,W),np.clip(cy+bh/2,1,H)],1),device='cuda',dtype=torch.float32)
    rois=torch.cat([torch.arange(8,device='cuda').repeat_interleave(n)[:,None],expanded.repeat(8,1)],1)
    return roi_align(ft,rois,output_size=(224,224),spatial_scale=1.,aligned=True).view(8,n,3,224,224).permute(1,0,2,3,4).contiguous()


@torch.inference_mode()
def extract(encoder,images,boxes,key_t=3,batch_size=8):
    ft=normalize_clip(images);parts=[]
    for j in range(0,len(boxes),batch_size):
        parts.append(encoder(crop_batch(ft,boxes[j:j+batch_size]),key_t).mean(1).half().cpu())
    crop=torch.cat(parts) if parts else torch.empty(0,1408,dtype=torch.float16)
    fm=encoder(F.interpolate(ft,size=(224,224),mode='bilinear',align_corners=False)[None],key_t).reshape(1,16,16,1408).permute(0,3,1,2)
    b=torch.tensor(boxes,device='cuda',dtype=torch.float32).reshape(-1,4)
    rois=torch.cat([b.new_zeros(len(b),1),b*16],1)
    context=roi_align(fm,rois,output_size=7,aligned=True).mean((-1,-2)).half().cpu()
    scene=F.adaptive_avg_pool2d(fm,(4,4)).flatten(2).transpose(1,2)[0].half().cpu()
    return {'crop':crop,'context_roi':context,'scene_tokens':scene,'boxes':b.cpu()}
