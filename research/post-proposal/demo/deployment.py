"""Reusable GPU-resident CSEB inference; original weights and sampling protocol."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align

from export_sample import sha
from infer_video import VideoReader, write_json
from predict import blend_scores, detector_scores, load_experts
from runtime import load_encoder, normalize_clip, crop_batch


class FeatureEncoder:
    """Return the trained pre-pooling tokens; omit the unused CLIP pooler.

    Reuses verified module weights without changing the source model or hooks.
    Direct block.forward calls bypass the research loader's capture hook.
    """
    def __init__(self, verified, compile_blocks=False):
        self.model = verified.model
        if self.model.sep_pos_embed or any(b.use_fused_rmsnorm for b in self.model.blocks):
            raise ValueError('Deployment requires the verified joint-position, unfused checkpoint topology')
        self.blocks = [torch.compile(b.forward, fullgraph=True, dynamic=False)
                       if compile_blocks else b.forward for b in self.model.blocks]
        self.audit = dict(verified.audit, unused_clip_pooler_removed=True,
                          backend='inductor-blocks' if compile_blocks else 'eager')

    @torch.inference_mode()
    def __call__(self, clips, key_t):
        if not 0 <= key_t < 8:
            raise ValueError('key_t must be in [0,7]')
        m = self.model
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = m.patch_embed(clips.permute(0,2,1,3,4).to(m.dtype))
            batch, temporal, spatial, channels = x.shape
            x = x.reshape(batch, temporal*spatial, channels)
            x = torch.cat((m.cls_token.expand(batch,-1,-1), x), dim=1) + m.pos_embed
            for block in self.blocks:
                x = block(x)
        if x.shape != (len(clips),2049,1408):
            raise ValueError('Unexpected encoder token shape')
        return x[:,1:].reshape(len(clips),8,256,1408)[:,key_t].float()


@torch.inference_mode()
def extract_gpu(encoder, images, boxes, key_t=3, batch_size=8):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1,4)
    if not np.isfinite(boxes).all() or (boxes<0).any() or (boxes>1).any() or (boxes[:,2:]<boxes[:,:2]).any():
        raise ValueError('Invalid normalized xyxy boxes')
    ft = normalize_clip(images)
    parts = [encoder(crop_batch(ft,boxes[j:j+batch_size]),key_t).mean(1).half()
             for j in range(0,len(boxes),batch_size)]
    crop = torch.cat(parts) if parts else ft.new_empty((0,1408),dtype=torch.float16)
    fm = encoder(F.interpolate(ft,size=(224,224),mode='bilinear',align_corners=False)[None],key_t)
    fm = fm.reshape(1,16,16,1408).permute(0,3,1,2)
    b = torch.as_tensor(boxes,device='cuda')
    rois = torch.cat((b.new_zeros((len(b),1)), b*16), dim=1)
    context = roi_align(fm,rois,output_size=7,aligned=True).mean((-1,-2)).half()
    scene = F.adaptive_avg_pool2d(fm,(4,4)).flatten(2).transpose(1,2)[0].half()
    result = {'crop':crop,'context_roi':context,'scene_tokens':scene,'boxes':b}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite visual features')
    return result


@torch.inference_mode()
def score_features(models, weight, features, confidence, batch_size=64):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    n = len(features['boxes'])
    q = torch.as_tensor(confidence,device='cuda',dtype=torch.float32)
    if q.shape != (n,) or not torch.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise ValueError('Invalid detector confidence')
    scores = [[],[]]
    for j in range(0,n,batch_size):
        end = min(j+batch_size,n)
        xs = [features['crop'][j:end],features['context_roi'][j:end],
              features['scene_tokens'][None].expand(end-j,-1,-1),features['boxes'][j:end]]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for k,model in enumerate(models):
                scores[k].append(model(*xs)['logits'].float().sigmoid())
    a,b = [torch.cat(s) if s else q.new_empty((0,184)) for s in scores]
    mixed = blend_scores(a,b,weight)
    result = {'classification':detector_scores(a,q),'contrastive':detector_scores(b,q),
              'blend':detector_scores(mixed,q)}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite scores')
    return result


class CSEBInference:
    """Load once, call predict_clip repeatedly, keep intermediates on the GPU."""
    def __init__(self, checkpoint, encoder_root, yolo, settings_dir,
                 crop_batch=8, compile_blocks=False, detector_conf=.25):
        if crop_batch<1 or not 0<detector_conf<1:
            raise ValueError('Invalid batch size/confidence')
        os.environ['YOLO_CONFIG_DIR'] = str(settings_dir)
        from ultralytics import YOLO
        self.labels=json.loads(Path(__file__).with_name('labels.json').read_text())
        self.detector=YOLO(str(yolo))
        if [self.detector.names[i] for i in range(10)]!=self.labels['agent']:
            raise ValueError('YOLO label ordering mismatch')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        signature=ck['signature'];del ck
        manifest={'source_type':'raw-video','protocol_sha256':signature['protocol_sha256'],
                  'training_cache_marker_sha256':signature['cache_marker_sha256']}
        model_path=Path(__file__).resolve().parents[1]/'contextual-roi-1b/model.py'
        self.models,self.weight,self.signature=load_experts(checkpoint,model_path,manifest,'cuda')
        self.encoder=FeatureEncoder(load_encoder(encoder_root),compile_blocks)
        self.crop_batch=crop_batch;self.detector_conf=detector_conf

    @torch.inference_mode()
    def predict_clip(self, images, key_t, return_features=False):
        torch.cuda.synchronize();start=time.perf_counter()
        result=self.detector.predict(images[key_t][:,:,::-1].copy(),imgsz=1280,
                  conf=self.detector_conf,iou=.7,max_det=300,device=0,half=True,verbose=False)[0]
        boxes=result.boxes.xyxyn.cpu().numpy().astype(np.float32)
        confidence=result.boxes.conf.float()
        agents=result.boxes.cls.long()
        torch.cuda.synchronize();detected=time.perf_counter()
        features=extract_gpu(self.encoder,images,boxes,key_t,self.crop_batch)
        torch.cuda.synchronize();encoded=time.perf_counter()
        scores=score_features(self.models,self.weight,features,confidence)
        torch.cuda.synchronize();ended=time.perf_counter()
        output={'boxes':boxes,'confidence':confidence.cpu().numpy(),'agents':agents.cpu().numpy(),
                'scores':{k:v.cpu().numpy() for k,v in scores.items()},
                'timings':{'yolo_seconds':detected-start,'encoder_seconds':encoded-detected,
                           'heads_seconds':ended-encoded,'gpu_pipeline_seconds':ended-start}}
        if return_features:
            output['features']={k:v.cpu() for k,v in features.items()}
        return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['video','checkpoint','encoder-root','yolo','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--frames',type=int,default=120)
    ap.add_argument('--start-frame',type=int,default=1)
    ap.add_argument('--crop-batch',type=int,default=8)
    ap.add_argument('--detector-conf',type=float,default=.25)
    ap.add_argument('--compile-blocks',action='store_true',help='Experimental; requires separate prediction validation')
    ap.add_argument('--save-features',action='store_true',help='Optional CPU feature exports for auditing')
    a=ap.parse_args()
    if a.frames<1 or a.start_frame<1 or a.crop_batch<1 or not 0<a.detector_conf<1:
        ap.error('Invalid frame, batch or confidence argument')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    a.output.mkdir(parents=True,exist_ok=False)
    settings=a.output/'yolo-settings';settings.mkdir()
    if a.save_features:(a.output/'features').mkdir()
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    write_json(a.output/'progress.json',{'phase':'loading','time':time.time()})
    runner=CSEBInference(a.checkpoint,a.encoder_root,a.yolo,settings,a.crop_batch,a.compile_blocks,a.detector_conf)
    loaded=time.perf_counter();reader=VideoReader(a.video)
    timings=[];candidates=0
    try:
        if a.start_frame>reader.sample_count:raise ValueError('Start frame exceeds video')
        with (a.output/'predictions.jsonl').open('w') as stream:
            for fid in range(a.start_frame,min(a.start_frame+a.frames,reader.sample_count+1)):
                start=time.perf_counter();images,fids,key_t=reader.clip(fid)
                decoded=time.perf_counter()
                result=runner.predict_clip(images,key_t,a.save_features)
                if a.save_features:torch.save(result.pop('features'),a.output/'features'/f'{fid:05d}.pt')
                record={'fid':fid,'input_fids':fids,'key_t':key_t,'boxes':result['boxes'].tolist(),
                        'confidence':result['confidence'].tolist(),'agents':result['agents'].tolist(),
                        'scores':{k:v.tolist() for k,v in result['scores'].items()}}
                stream.write(json.dumps(record,separators=(',',':'))+'\n');stream.flush()
                timing={'fid':fid,'candidates':len(result['boxes']),'decode_seconds':decoded-start,
                        **result['timings'],'frame_wall_seconds':time.perf_counter()-start}
                timings.append(timing);candidates+=len(result['boxes'])
                progress={'phase':'inference','frames':len(timings),'time':time.time(),'last_frame':timing}
                write_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    finally:reader.cap.release()
    report={'frames':len(timings),'candidates':candidates,'load_seconds':loaded-started,
            'total_wall_seconds':time.perf_counter()-started,'frame_timings':timings,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'encoder':runner.encoder.audit,'weight':runner.weight,'labels':runner.labels,
            'crop_batch':a.crop_batch,'features_saved':a.save_features,
            'detector':{'confidence':a.detector_conf,'imgsz':1280,'iou':.7,'max_det':300},
            'temporal_note':'Unchanged offline eight-frame window, keyframe-3 through keyframe+4 at10Hz.',
            'source_sha256':{name:sha(Path(__file__).with_name(name)) for name in ['deployment.py','runtime.py','infer_video.py','predict.py']},
            'inputs_sha256':{name:sha(getattr(a,name)) for name in ['video','checkpoint','yolo']}}
    write_json(a.output/'report.json',report)
    write_json(a.output/'progress.json',{'phase':'complete','frames':len(timings),'time':time.time()})
    print(json.dumps({'complete':True,'output':str(a.output),'seconds':report['total_wall_seconds']}),flush=True)


if __name__=='__main__':main()
