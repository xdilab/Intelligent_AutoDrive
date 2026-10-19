"""Bounded paired benchmark of original and GPU-resident CSEB execution."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from deployment import FeatureEncoder, extract_gpu, score_features
from export_sample import sha
from infer_video import VideoReader, write_json
from predict import load_experts
from runtime import load_encoder, extract, crop_batch, normalize_clip


def compare(actual, expected):
    result={}
    for key in actual:
        a,b=actual[key].cpu(),expected[key].cpu()
        delta=(a-b).abs()
        result[key]={'max_abs_error':float(delta.max()) if delta.numel() else 0,
                     'mean_abs_error':float(delta.mean()) if delta.numel() else 0,
                     'allclose':bool(torch.allclose(a,b,atol=1e-4,rtol=1e-3)),
                     'triplet_top1_equal':bool(torch.equal(a[:,98:].argmax(1),b[:,98:].argmax(1)))}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['video','checkpoint','encoder-root','yolo','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--frames',type=int,default=3)
    ap.add_argument('--start-frame',type=int,default=7)
    ap.add_argument('--include-compile',action='store_true')
    ap.add_argument('--compile-only',action='store_true',help='Compare reference and compiler only')
    a=ap.parse_args()
    if a.frames<1:ap.error('frames must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'yolo-settings').mkdir()
    os.environ['YOLO_CONFIG_DIR']=str(a.output/'yolo-settings')
    from ultralytics import YOLO
    torch.set_num_threads(8)
    started=time.perf_counter()
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'start_frame':a.start_frame,'frames':a.frames,'variants':[],
            'acceptance':{'atol':1e-4,'rtol':1e-3,'triplet_top1_equal':True},
            'scope':'Same freshly decoded frames and YOLO proposals for all variants. Timings exclude decode/detector, JSON export, and warmup. No AP evaluation.'}
    def progress(phase,**kwargs):
        record={'phase':phase,'time':time.time(),'elapsed_seconds':time.perf_counter()-started,**kwargs}
        write_json(a.output/'progress.json',record)
        write_json(a.output/'report.json',report)
        print(json.dumps(record),flush=True)
    progress('loading')
    encoder=load_encoder(a.encoder_root)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    signature=ck['signature'];del ck
    manifest={'source_type':'raw-video','protocol_sha256':signature['protocol_sha256'],
              'training_cache_marker_sha256':signature['cache_marker_sha256']}
    model_path=Path(__file__).resolve().parents[1]/'contextual-roi-1b/model.py'
    models,weight,_=load_experts(a.checkpoint,model_path,manifest,'cuda')
    detector=YOLO(str(a.yolo));reader=VideoReader(a.video);frames=[]
    try:
        for fid in range(a.start_frame,a.start_frame+a.frames):
            images,fids,key=reader.clip(fid)
            r=detector.predict(images[key][:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0]
            frames.append((images,r.boxes.xyxyn.cpu().numpy().astype(np.float32),key,r.boxes.conf.float()))
    finally:reader.cap.release()
    report['candidates']=[len(f[1]) for f in frames]
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        encoder(crop_batch(normalize_clip(frames[0][0]),frames[0][1][:1]),frames[0][2])
        torch.cuda.synchronize()
    report['attention_operators']=[{'name':event.key,'count':event.count} for event in prof.key_averages()
                                   if any(term in event.key.lower() for term in ['flash_attention','efficient_attention','scaled_dot_product'])]
    del prof
    expected=[]
    variants=[('reference',8,encoder)]
    if not a.compile_only:
        variants.extend([('resident',8,FeatureEncoder(encoder)),('resident',16,FeatureEncoder(encoder))])
    if a.include_compile or a.compile_only:variants.append(('compiled',8,FeatureEncoder(encoder,True)))
    for name,batch,engine in variants:
        record={'name':name,'batch':batch,'timings':[],'comparisons':[]}
        report['variants'].append(record)
        def run(frame):
            images,boxes,key,q=frame
            if name=='reference':
                features={k:v.cuda() for k,v in extract(engine,images,boxes,key,batch).items()}
            else:features=extract_gpu(engine,images,boxes,key,batch)
            return score_features(models,weight,features,q)
        try:
            progress('warmup',variant=name,batch=batch)
            torch.cuda.synchronize();t=time.perf_counter();run(frames[0]);torch.cuda.synchronize()
            record['warmup_seconds']=time.perf_counter()-t
            torch.cuda.reset_peak_memory_stats()
            for i,frame in enumerate(frames):
                progress('measuring',variant=name,batch=batch,frame=i+1)
                torch.cuda.synchronize();t=time.perf_counter();scores=run(frame);torch.cuda.synchronize()
                record['timings'].append(time.perf_counter()-t)
                if name=='reference':expected.append({k:v.cpu() for k,v in scores.items()})
                else:record['comparisons'].append(compare(scores,expected[i]))
            record['median_seconds']=statistics.median(record['timings'])
            record['total_seconds']=sum(record['timings'])
            record['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
            record['passed']=all(v['allclose'] and v['triplet_top1_equal'] for c in record['comparisons'] for v in c.values())
            record['speedup']=report['variants'][0]['total_seconds']/record['total_seconds']
            progress('variant-complete',variant=name,batch=batch,passed=record['passed'],seconds=record['total_seconds'],speedup=record['speedup'])
        except Exception as exc:
            record.update(passed=False,error=f'{type(exc).__name__}: {exc}')
            progress('variant-failed',variant=name,batch=batch,error=record['error'])
            if name=='reference':raise
            torch.cuda.empty_cache()
    report['source_sha256']={name:sha(Path(__file__).with_name(name)) for name in ['deployment.py','benchmark_deployment.py','runtime.py']}
    report['checkpoint_sha256']=sha(a.checkpoint)
    report['complete']=True
    progress('complete')


if __name__=='__main__':main()
