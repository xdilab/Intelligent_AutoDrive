"""Portable causal raw-video runner. Requires only the files in its bundle."""
import argparse,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
class Runner:
 def __init__(self,bundle,settings):
  self.bundle=Path(bundle).resolve();m=json.loads((self.bundle/'bundle.json').read_text());self.labels=m['labels']
  for name,h in m['files_sha256'].items():
   if Path(name).name!=name or sha(self.bundle/name)!=h:raise ValueError('Bundle integrity failure: '+name)
  spec=importlib.util.spec_from_file_location('portable_student_model',self.bundle/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  self.model=module.Student().cuda().eval();self.model.load_state_dict(torch.load(self.bundle/'student.pt',map_location='cpu',weights_only=True));self.backbone=module.SharedBackbone(self.bundle/'backbone.pth').cuda()
  os.environ['YOLO_CONFIG_DIR']=str(settings);Path(settings).mkdir(parents=True,exist_ok=True)
  from ultralytics import YOLO
  self.detector=YOLO(str(self.bundle/'yolo.pt'));assert [self.detector.names[i] for i in range(10)]==self.labels['agent'];self.reset()
 def reset(self):self.state=None
 @torch.inference_mode()
 def predict_frame(self,image):
  d=self.detector.predict(image[:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0];box=d.boxes.xyxyn.float();agents=d.boxes.cls.long();q=d.boxes.conf.float()
  roi,scene=self.backbone(image,box);z,self.state=self.model(roi.half().float(),scene.half().float(),box,agents,self.state);scores=z.sigmoid()*q[:,None];scores[:,0]=q
  assert torch.isfinite(scores).all() and torch.isfinite(self.state).all()
  return dict(boxes=box.cpu().numpy(),agents=agents.cpu().numpy(),confidence=q.cpu().numpy(),scores=scores.cpu().numpy())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path(__file__).parent);ap.add_argument('--video',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=0);ap.add_argument('--verify-isolation',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8)
 opened=[]
 if a.verify_isolation:
  def audit(event,args):
   if event=='open' and isinstance(args[0],(str,bytes)):
    path=Path(os.fsdecode(args[0])).resolve()
    if path.suffix in ['.pt','.pth','.safetensors','.npy','.npz','.pkl']:
     if path.parent!=a.bundle.resolve():raise RuntimeError('External model/cache asset forbidden: '+str(path))
     opened.append(str(path))
  sys.addaudithook(audit)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 begin=time.perf_counter();runner=Runner(a.bundle,a.output/'yolo-settings');torch.cuda.synchronize();load=time.perf_counter()-begin;cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert cap.isOpened() and fps>0 and count>0
 samples=math.floor((count-1)*10/fps)+1;limit=min(samples,a.frames) if a.frames else samples;next_index=0;records=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 image=None
 with (a.output/'predictions.jsonl').open('w') as out:
  for fid in range(1,limit+1):
   t=time.perf_counter();index=min(count-1,round((fid-1)*fps/10))
   while next_index<=index:
    ok,image=cap.read();assert ok;next_index+=1
   assert next_index-1==index # Never decode a future source frame.
   rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);result=runner.predict_frame(rgb);torch.cuda.synchronize();out.write(json.dumps(dict(fid=fid,source_index=index,**{k:v.tolist() for k,v in result.items()}),separators=(',',':'))+'\n');out.flush()
   records.append(dict(fid=fid,source_index=index,candidates=len(result['boxes']),seconds=time.perf_counter()-t,gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved(),cpu_rss_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')))
 cap.release();report=dict(frames=limit,source_video=str(a.video),source_sha256=sha(a.video),fps=fps,sampling_hz=10,load_seconds=load,processing_seconds=time.perf_counter()-start,records=records,causal_source_indices_verified=True,external_assets_blocked=a.verify_isolation,opened_model_assets=sorted(set(opened)),peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,bundle_sha256=sha(a.bundle/'bundle.json'),labels=runner.labels)
 (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['records','labels']}),flush=True)
if __name__=='__main__':main()
