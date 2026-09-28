"""Extract one frozen full-scene clip per frame key; cache shared spatial context + actor RoIs.
Input JSONL uses the established stage56-full frame/box manifest (no new box sampling).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align

MODEL = 'OpenGVLab/InternVideo2_CLIP_S'
REVISION = '1f9fca1389fd883defc652634d95a21121c85a8c'
CONTRACT = {'version':1, 'model':MODEL, 'revision':REVISION,
            'resize':'full frame anisotropic bilinear224x224, align_corners=False; no center crop',
            'normalization':'ImageNet mean/std before resize',
            'feature':'last-block keyframe256x1024 tokens', 'roi':'un-padded normalized box;7x7 RoIAlign aligned=True then mean',
            'scene':'4x4 adaptive average pooling;16x1024 shared tokens per frame key'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def contextual_features(feature_map, boxes):
    """feature_map1,D,H,W and normalized xyxy; geometry matches full-frame resize."""
    if feature_map.ndim != 4 or feature_map.shape[0] != 1:
        raise ValueError('Expected one full-frame spatial map')
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or not torch.isfinite(boxes).all():
        raise ValueError('Expected finite Nx4 boxes')
    if (boxes<0).any() or (boxes>1).any() or (boxes[:,2:]<boxes[:,:2]).any():
        raise ValueError('Invalid normalized boxes')
    # Preserve zero-area boundary candidates; aligned RoIAlign yields zero context.
    # Do not remove/reorder candidates or enlarge their original boxes.
    h,w = feature_map.shape[-2:]
    coords = boxes * boxes.new_tensor([w,h,w,h])
    rois = torch.cat([boxes.new_zeros((len(boxes),1)),coords],1)
    actor = roi_align(feature_map,rois,output_size=7,spatial_scale=1,aligned=True).mean((-1,-2))
    scene = F.adaptive_avg_pool2d(feature_map,(4,4)).flatten(2).transpose(1,2)[0]
    return actor,scene

class FrozenSceneEncoder:
    def __init__(self,device):
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cfg=AutoConfig.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True,local_files_only=True)
        cls=get_class_from_dynamic_module('modeling_internvideo2encoder.InternVideo2_CLIP_small',MODEL,revision=REVISION,local_files_only=True)
        self.model=cls(cfg)
        self.model.load_state_dict(load_file(hf_hub_download(MODEL,'model.safetensors',revision=REVISION,local_files_only=True)),strict=True)
        self.model.requires_grad_(False).eval().to(device)
        self.model.vision_encoder.blocks[-1].register_forward_hook(self.capture)
        self.device=device
        self.tokens=None

    def capture(self,module,inputs,output):
        self.tokens=output if isinstance(output,torch.Tensor) else output[0]

    @torch.inference_mode()
    def __call__(self,row,frames):
        arrays=[]
        if len(row['fids'])!=8 or len(row['frame_sha256'])!=8:
            raise ValueError('Eight frames and hashes required')
        for fid,sha in zip(row['fids'],row['frame_sha256']):
            path=frames/row['video']/f'{fid:05d}.jpg'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError(f'Frame digest changed: {path}')
            with Image.open(path) as im:arrays.append(np.array(im.convert('RGB')))
        x=torch.tensor(np.stack(arrays),device=self.device).permute(0,3,1,2).float()/255
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        x=F.interpolate((x-mean)/std,size=(224,224),mode='bilinear',align_corners=False)[None]
        self.tokens=None
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            self.model.encode_vision(x)
        tok=self.tokens
        if tok is None or tok.shape[-1]!=1024 or tok.shape[1] not in (2048,2049):
            raise ValueError(f'Unexpected encoder token layout: {None if tok is None else tok.shape}')
        self.tokens=None
        fmap=tok[:,-2048:].reshape(1,8,16,16,1024)[:,row['key_t']].permute(0,3,1,2).float()
        boxes=torch.tensor(row['boxes'],device=self.device,dtype=torch.float32).reshape(-1,4)
        roi,scene=contextual_features(fmap,boxes)
        if not torch.isfinite(roi).all() or not torch.isfinite(scene).all():raise ValueError('Nonfinite cache')
        return {'boxes':boxes.cpu(),'context_roi':roi.half().cpu(),'scene_tokens':scene.half().cpu()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--frames',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--device',default='cuda');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1);ap.add_argument('--shard',type=int,default=0)
    args=ap.parse_args();assert 0<=args.shard<args.shards;args.output.mkdir(parents=True,exist_ok=True)
    encoder=None;completed=0
    for row_index,line in enumerate(args.manifest.open()):
        if row_index%args.shards!=args.shard:continue
        row=json.loads(line);key=f"{row['video']}_{int(row['fid']):05d}"
        # Safe file names even for externally supplied manifests.
        if '/' in key or '\\' in key or '..' in key:raise ValueError(key)
        fingerprint=digest({'contract':CONTRACT,'row':row});dest=args.output/f'{key}.pt'
        if dest.exists():
            saved=torch.load(dest,map_location='cpu',weights_only=True)
            if saved['fingerprint']!=fingerprint:raise ValueError(f'Cache mismatch: {dest}')
        else:
            if encoder is None:encoder=FrozenSceneEncoder(torch.device(args.device))
            data=encoder(row,args.frames)
            data.update(fingerprint=fingerprint,contract=CONTRACT,key=key,row_sha256=digest(row))
            tmp=dest.with_suffix('.pt.tmp');torch.save(data,tmp);tmp.replace(dest)
        completed+=1
        status={'cached':completed,'key':key,'resumable':True,'time':time.time(),'manifest':str(args.manifest),'shard':args.shard}
        heartbeat=args.output/f'progress-{args.shard}.json';tmp=heartbeat.with_suffix('.tmp');tmp.write_text(json.dumps(status));tmp.replace(heartbeat)
        if completed%100==0:print(json.dumps(status),flush=True)
        if args.limit and completed>=args.limit:break

if __name__=='__main__':main()
