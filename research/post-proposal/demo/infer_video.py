"""End-to-end raw-video CSEB inference: fresh YOLO, fresh 1B features, experts, video.

No dataset labels, precomputed proposals, or precomputed visual features are used.
The fixed trained phrase bank is embedded in the checkpoint.
"""
import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

from export_sample import sha
from runtime import extract,load_encoder


class VideoReader:
    def __init__(self,path):
        self.cap=cv2.VideoCapture(str(path))
        self.fps=self.cap.get(cv2.CAP_PROP_FPS)
        self.count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.cap.isOpened() or self.fps<=0 or self.count<=0:
            raise ValueError('Video must be readable with valid timing metadata')
        self.sample_count=max(1,math.floor((self.count-1)*10/self.fps)+1)
        self.cache=OrderedDict();self.next_index=0

    def frame(self,fid):
        index=min(self.count-1,round((fid-1)*self.fps/10))
        if index in self.cache:return self.cache[index]
        if index<self.next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,index);self.next_index=index
        image=None
        while self.next_index<=index:
            ok,image=self.cap.read();self.next_index+=1
            if not ok:raise ValueError(f'Video decode failed at {self.next_index-1}')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.cache[index]=image
        while len(self.cache)>32:self.cache.popitem(last=False)
        return image

    def clip(self,fid):
        fids=[min(max(fid-3+j,1),self.sample_count) for j in range(8)]
        return np.stack([self.frame(f) for f in fids]),fids,fids.index(fid)


def write_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--encoder-root',type=Path,required=True,help='Contains results/assets.json and exact base/CLIP weights')
    ap.add_argument('--yolo',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=120)
    ap.add_argument('--start-frame',type=int,default=1,help='1-based frame in the resampled 10 Hz input')
    ap.add_argument('--detector-conf',type=float,default=.25)
    ap.add_argument('--crop-batch',type=int,default=8)
    ap.add_argument('--extract-only',action='store_true',help='Benchmark/inspect fresh features without rendering')
    a=ap.parse_args()
    if a.frames<1 or a.start_frame<1 or a.crop_batch<1 or not 0<a.detector_conf<1:
        ap.error('Invalid frame/count/confidence argument')
    if not torch.cuda.is_available():raise RuntimeError('End-to-end 1B inference requires an accessible CUDA GPU')
    a.output.mkdir(parents=True,exist_ok=False)
    sample=a.output/'sample';sample.mkdir();(sample/'cache').mkdir();(sample/'frames').mkdir()
    # Keep detector settings in the run directory, not a global research environment.
    (a.output/'yolo-settings').mkdir()
    os.environ['YOLO_CONFIG_DIR']=str(a.output/'yolo-settings')
    from ultralytics import YOLO
    torch.set_num_threads(8)
    labels=json.loads(Path(__file__).with_name('labels.json').read_text())
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('kind')!='blend':raise ValueError('Expected the scaled two-expert checkpoint')
    signature=ck['signature'];del ck
    started=time.monotonic();reader=VideoReader(a.video)
    if a.start_frame>reader.sample_count:raise ValueError('Start is past end of video')
    end=min(a.start_frame+a.frames,reader.sample_count+1)
    write_json(a.output/'progress.json',{'phase':'loading','time':time.time(),'total':end-a.start_frame})
    detector=YOLO(str(a.yolo))
    if [detector.names[i] for i in range(10)]!=labels['agent']:
        raise ValueError('YOLO agent class order differs from trained ROAD model')
    encoder=load_encoder(a.encoder_root)
    load_seconds=time.monotonic()-started
    records=[];confidences=[];agent_classes=[];offset=0;timings=[]
    try:
        for fid in range(a.start_frame,end):
            t=time.monotonic();images,fids,key_t=reader.clip(fid)
            # Ultralytics ndarray input is BGR, the encoder input is RGB.
            result=detector.predict(images[key_t][:,:,::-1].copy(),imgsz=1280,conf=a.detector_conf,
                                    iou=.7,max_det=300,device=0,half=True,verbose=False)[0]
            boxes=result.boxes.xyxyn.cpu().numpy().astype(np.float32)
            q=result.boxes.conf.cpu().numpy().astype(np.float32);agents=result.boxes.cls.cpu().numpy().astype(np.int64)
            torch.cuda.synchronize();detection_seconds=time.monotonic()-t
            features=extract(encoder,images,boxes,key_t,a.crop_batch)
            torch.cuda.synchronize();total_seconds=time.monotonic()-t
            key=f'input_{fid:05d}';cache_path=sample/'cache'/f'{key}.pt'
            torch.save(features,cache_path)
            image_path=sample/'frames'/f'{key}.jpg';Image.fromarray(images[key_t]).save(image_path,quality=95)
            records.append({'key':key,'video':a.video.stem,'fid':fid,'row_start':offset,'row_end':offset+len(boxes),
                            'cache_sha256':sha(cache_path),'image_sha256':sha(image_path),
                            'input_fids':fids,'key_t':key_t})
            offset+=len(boxes);confidences.append(q);agent_classes.append(agents)
            timings.append({'fid':fid,'candidates':len(boxes),'decode_and_yolo_seconds':detection_seconds,
                            'visual_encoder_seconds':total_seconds-detection_seconds,'total_seconds':total_seconds})
            progress={'phase':'fresh-feature-extraction','frames':len(records),'total':end-a.start_frame,
                      'candidates':offset,'seconds':time.monotonic()-started,'last_frame':timings[-1],'time':time.time()}
            write_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    finally:reader.cap.release()
    np.save(sample/'q.npy',np.concatenate(confidences));np.save(sample/'agent.npy',np.concatenate(agent_classes))
    manifest={'format':1,'source_type':'raw-video','split':'user-input','labels':labels,'frames':records,
              'selection':'contiguous input video segment; no prediction-based selection',
              'protocol_sha256':signature['protocol_sha256'],
              'training_cache_marker_sha256':signature['cache_marker_sha256'],
              'q_sha256':sha(sample/'q.npy'),'agent_sha256':sha(sample/'agent.npy'),
              'video_sha256':sha(a.video),'yolo_sha256':sha(a.yolo),'source_fps':reader.fps,'sampling_fps':10,
              'detector':{'confidence':a.detector_conf,'imgsz':1280,'iou':.7,'max_det':300},
              'encoder_assets':json.loads((a.encoder_root/'results/assets.json').read_text()),
              'encoder_audit':encoder.audit,'load_seconds':load_seconds,'frame_timings':timings,
              'implementation_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['infer_video.py','runtime.py']},
              'temporal_note':'8 frames at 10 Hz: keyframe minus3 through plus4; offline/noncausal, with endpoint repetition.'}
    write_json(sample/'manifest.json',manifest)
    del encoder,detector;torch.cuda.empty_cache()
    if not a.extract_only:
        write_json(a.output/'progress.json',{'phase':'expert-inference','time':time.time()})
        subprocess.run([sys.executable,str(Path(__file__).with_name('predict.py')),'--sample',str(sample),
                        '--checkpoint',str(a.checkpoint),'--output',str(a.output/'predictions'),'--device','cuda'],check=True)
        write_json(a.output/'progress.json',{'phase':'rendering','time':time.time()})
        subprocess.run([sys.executable,str(Path(__file__).with_name('render.py')),'--sample',str(sample),
                        '--predictions',str(a.output/'predictions/predictions.json'),
                        '--output',str(a.output/'viewer'),'--mp4'],check=True)
    write_json(a.output/'progress.json',{'phase':'extraction-complete' if a.extract_only else 'complete',
               'frames':len(records),'candidates':offset,'total_wall_seconds':time.monotonic()-started,'time':time.time()})
    print(json.dumps({'complete':True,'frames':len(records),'output':str(a.output),'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
