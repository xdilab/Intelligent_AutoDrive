"""Evaluation-only input reuse; unchanged crops, batch size, encoder and scoring."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO
import hashlib,time,json
import numpy as np
import torch
from PIL import Image
from torchvision.ops import roi_align
import base

class ClipLoader:
    def __init__(self,cfg,capacity=24):
        self.root=base.frame_root(cfg);self.cache=OrderedDict();self.capacity=capacity
        self.reads=0;self.hits=0
    def load(self,row):
        imgs=[];digests=[]
        for fid in row['fids']:
            path=self.root/row['video']/f'{fid:05d}.jpg'
            if path in self.cache:
                arr,digest=self.cache.pop(path);self.hits+=1
            else:
                raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                with Image.open(BytesIO(raw)) as im:arr=np.asarray(im.convert('RGB')).copy()
                self.reads+=1
            self.cache[path]=(arr,digest)
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
            imgs.append(arr);digests.append(digest)
        if 'frame_sha256' in row:assert digests==row['frame_sha256']
        return np.stack(imgs)

def normalize_clip(imgs):
    ft=torch.from_numpy(imgs).cuda().permute(0,3,1,2).float()/255
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    return (ft-mean)/std

def crop_batch(ft,boxes):
    H,W=ft.shape[-2:];b=np.array(boxes);n=len(b)
    cx=(b[:,0]+b[:,2])/2*W;cy=(b[:,1]+b[:,3])/2*H
    bw=np.maximum((b[:,2]-b[:,0])*W,8)*2;bh=np.maximum((b[:,3]-b[:,1])*H,8)*2
    boxes=torch.tensor(np.stack([np.clip(cx-bw/2,0,W-1),np.clip(cy-bh/2,0,H-1),np.clip(cx+bw/2,1,W),np.clip(cy+bh/2,1,H)],1),device='cuda',dtype=torch.float32)
    rois=torch.cat([torch.arange(8,device='cuda').repeat_interleave(n)[:,None],boxes.repeat(8,1)],1)
    x=roi_align(ft,rois,output_size=(224,224),spatial_scale=1.,aligned=True).view(8,n,3,224,224).permute(1,0,2,3,4).contiguous()
    assert x.shape==(n,8,3,224,224) and torch.isfinite(x).all()
    return x

def rows(keys,yolo,ann,pred):
    for i,s in enumerate(keys):
        if (pred/f'{s}.npy').exists():continue
        v,f=s.rsplit('_',1);fid=int(f);nf=ann['db'][v]['numf']
        fids=[min(max(fid-3+j,1),nf) for j in range(8)]
        yield i,s,{'video':v,'fid':fid,'fids':fids,'key_t':fids.index(fid),'boxes':yolo[s]['boxes_xyxyn'].tolist()}

def prefetched(source,loader):
    source=iter(source)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='eval-decode') as pool:
        current=next(source,None)
        if current is None:return
        future=pool.submit(loader.load,current[2])
        while current is not None:
            imgs=future.result();following=next(source,None)
            if following is not None:future=pool.submit(loader.load,following[2])
            yield *current,imgs
            current=following

@torch.no_grad()
def predictions(m,keys,yolo,ann,cfg,pred,batch_size=64):
    loader=ClipLoader(cfg);start=time.monotonic();processed=0;crops=0
    for i,s,row,imgs in prefetched(rows(keys,yolo,ann,pred),loader):
        scores=[]
        if len(row['boxes']):
            ft=normalize_clip(imgs)
            for j in range(0,len(row['boxes']),batch_size):
                x=crop_batch(ft,row['boxes'][j:j+batch_size])
                with torch.autocast('cuda',dtype=torch.bfloat16):z,_,_=m(x,row['key_t'])
                scores.append(z.float().sigmoid().cpu().numpy())
                del x,z
            del ft
        sig=np.concatenate(scores) if scores else np.empty((0,184),np.float32)
        processed+=1;crops+=len(sig)
        yield i,s,sig
        if processed==1 or processed%25==0:
            print('INPUT_REUSE',json.dumps({'frames':processed,'crops':crops,'seconds':time.monotonic()-start,'jpeg_reads':loader.reads,'jpeg_cache_hits':loader.hits}),flush=True)
