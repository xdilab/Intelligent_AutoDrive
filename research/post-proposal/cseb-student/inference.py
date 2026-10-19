"""Causal student video inference; explicit reset at each new video."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from model import SharedBackbone,Student
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demo'))
from export_sample import sha
from infer_video import VideoReader,write_json
from predict import detector_scores

class StudentInference:
    def __init__(self,run,settings,condition='distilled'):
        if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
        run=Path(run)
        if (run/'bundle.json').is_file():
            manifest=json.loads((run/'bundle.json').read_text())
            if manifest['format']!=1:raise ValueError('Unsupported student bundle')
            if sha(Path(__file__).with_name('model.py'))!=manifest['model_sha256']:raise ValueError('Student model source mismatch')
            for name,digest in manifest['files_sha256'].items():
                if Path(name).name!=name or sha(run/name)!=digest:raise ValueError('Bundle asset mismatch: '+name)
            backbone=run/'backbone.pth';yolo=run/'yolo.pt'
            state=torch.load(run/'student.pt',map_location='cpu',weights_only=True)
        else:
            manifest=json.loads((run/'manifest.json').read_text())
            assert sha(Path(__file__).with_name('model.py'))==manifest['source_sha256']['model.py']
            backbone=Path(manifest['backbone']);assert sha(backbone)==manifest['backbone_sha256']
            checkpoint=torch.load(run/f'{condition}-best.pt',map_location='cpu',weights_only=False)
            assert checkpoint['manifest_sha256']==sha(run/'manifest.json')
            state=checkpoint['model']
            yolo=Path('/data/repos/ROAD_Reason/experiments/exp11_yolo/runs/v8x_agent_1280_eccv/weights/best.pt')
        self.model=Student().cuda().eval();self.model.load_state_dict(state,strict=True)
        self.backbone=SharedBackbone(backbone).cuda()
        Path(settings).mkdir(parents=True,exist_ok=True)
        os.environ['YOLO_CONFIG_DIR']=str(settings)
        from ultralytics import YOLO
        self.detector=YOLO(str(yolo));self.labels=manifest['labels']
        if [self.detector.names[i] for i in range(10)]!=self.labels['agent']:raise ValueError('Detector label ordering mismatch')
        self.reset()
    def reset(self):self.state=None
    @torch.inference_mode()
    def predict_frame(self,image):
        det=self.detector.predict(image[:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0]
        boxes=det.boxes.xyxyn.float();agents=det.boxes.cls.long();q=det.boxes.conf.float()
        roi,scene=self.backbone(image,boxes)
        # Match the FP16 feature storage used for this frozen-backbone pilot.
        logits,self.state=self.model(roi.half().float(),scene.half().float(),boxes,agents,self.state)
        scores=detector_scores(logits.sigmoid(),q)
        assert torch.isfinite(scores).all()
        return {'boxes':boxes.cpu().numpy(),'agents':agents.cpu().numpy(),'confidence':q.cpu().numpy(),'scores':scores.cpu().numpy()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--video',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=120);a=ap.parse_args()
    if a.frames<1:ap.error('frames must be positive')
    a.output.mkdir(exist_ok=False,parents=True);settings=a.output/'yolo-settings';settings.mkdir()
    torch.set_num_threads(8);start=time.perf_counter();runner=StudentInference(a.run,settings);loaded=time.perf_counter();reader=VideoReader(a.video);count=0
    try:
        with (a.output/'predictions.jsonl').open('w') as f:
            for fid in range(1,min(a.frames,reader.sample_count)+1):
                out=runner.predict_frame(reader.frame(fid));f.write(json.dumps({'fid':fid,**{k:v.tolist() for k,v in out.items()}},separators=(',',':'))+'\n');count+=1
    finally:reader.cap.release()
    write_json(a.output/'report.json',{'frames':count,'load_seconds':loaded-start,'inference_and_decode_seconds':time.perf_counter()-loaded,'labels':runner.labels,'model':'experimental causal student; pilot validation only','temporal':'current/past frames only;10Hz; reset at video start'})
if __name__=='__main__':main()
