"""One architecture per process for clean load and GPU-memory measurements."""
from common import *
from standalone import Runner
from infer_video import VideoReader
from torch.nn import functional as F
from torchvision.ops import roi_align
import deployment
def sync():torch.cuda.synchronize();return time.perf_counter()
@torch.inference_mode()
def timed_student(runner,image):
 t=sync();d=runner.detector.predict(image[:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0];box=d.boxes.xyxyn.float();agent=d.boxes.cls.long();q=d.boxes.conf.float();dtime=sync()
 x=torch.as_tensor(image,device='cuda').permute(2,0,1)[None].float()/255;x=F.interpolate(x,size=(448,672),mode='bilinear',align_corners=False);mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None];x=(x-mean)/std;ptime=sync()
 with torch.autocast('cuda',dtype=torch.bfloat16):fmap=runner.backbone.body(x)
 fmap=fmap.float();rois=torch.cat((box.new_zeros((len(box),1)),box*box.new_tensor([42,28,42,28])),1);roi=roi_align(fmap,rois,output_size=3,aligned=True).mean((-1,-2));scene=fmap.mean((-1,-2))[0];etime=sync()
 z,runner.state=runner.model(roi.half().float(),scene.half().float(),box,agent,runner.state);scores=z.sigmoid()*q[:,None];scores[:,0]=q;htime=sync();assert torch.isfinite(scores).all()
 out=dict(boxes=box.cpu().numpy(),agents=agent.cpu().numpy(),confidence=q.cpu().numpy(),scores=scores.cpu().numpy());end=sync()
 return out,dict(detector_seconds=dtime-t,preprocessing_seconds=ptime-dtime,encoding_seconds=etime-ptime,head_seconds=htime-etime,cpu_export_seconds=end-htime)
def main():
 mode=sys.argv[1];assert mode in ['student','teacher'];setup();root=ROOT/'runtime'/mode;root.mkdir(parents=True,exist_ok=True);torch.cuda.reset_peak_memory_stats();t=sync()
 runner=Runner(ROOT/'bundle-seed0',root/'yolo-settings') if mode=='student' else p.CSEBInference(p.TEACHER,p.ENCODER,p.YOLO,root/'yolo-settings');load=sync()-t
 if mode=='student':params=dict(encoder=sum(v.numel() for v in runner.backbone.parameters()),heads=sum(v.numel() for v in runner.model.parameters()),detector=sum(v.numel() for v in runner.detector.model.parameters()))
 else:params=dict(encoder_loaded=sum(v.numel() for v in runner.encoder.model.parameters()),heads=sum(v.numel() for m in runner.models for v in m.parameters()),detector=sum(v.numel() for v in runner.detector.model.parameters()))
 preprocess=[]
 if mode=='teacher':
  def wrap(fn):
   def measured(*args,**kw):
    t=sync();out=fn(*args,**kw);preprocess.append(sync()-t);return out
   return measured
  deployment.normalize_clip=wrap(deployment.normalize_clip);deployment.crop_batch=wrap(deployment.crop_batch)
 video=read(ROOT/'split-manifest.json')['groups']['train'][0];path=p.ROOT/'videos'/f'{video}.mp4';reader=VideoReader(path);records=[]
 with (root/'predictions.jsonl').open('w') as output:
  for fid in range(1,37):
   t=sync();preprocess.clear()
   if mode=='student':image=reader.frame(fid);decoded=sync();out,times=timed_student(runner,image);input_fids=[fid]
   else:
    images,input_fids,key=reader.clip(fid);image=images[key];decoded=sync();raw=runner.predict_clip(images,key);endpredict=sync();out={k:raw[k] for k in ['boxes','agents','confidence']};out['scores']=raw['scores']['blend'];original=raw['timings'];pre=sum(preprocess)
    times=dict(detector_seconds=original['yolo_seconds'],preprocessing_seconds=pre,encoding_seconds=original['encoder_seconds']-pre,head_seconds=original['heads_seconds'],cpu_export_seconds=max(0,endpredict-decoded-original['gpu_pipeline_seconds']))
   serialized=time.perf_counter();output.write(json.dumps(dict(fid=fid,**{k:v.tolist() for k,v in out.items()}),separators=(',',':'))+'\n');output.flush();end=sync()
   candidate=hashlib.sha256(out['boxes'].tobytes()+out['agents'].tobytes()+out['confidence'].tobytes()).hexdigest()
   rec=dict(fid=fid,warmup=fid<=4,candidates=len(out['boxes']),candidate_sha256=candidate,input_fids=input_fids,resolution=list(image.shape[:2]),decode_seconds=decoded-t,**times,json_export_seconds=end-serialized,total_seconds=end-t,gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved());records.append(rec);progress('runtime-benchmark',architecture=mode,**rec)
 reader.cap.release();measured=[r for r in records if not r['warmup']];keys=[k for k in measured[0] if k.endswith('_seconds')]
 summary={k:dict(mean=float(np.mean([r[k] for r in measured])),median=float(np.median([r[k] for r in measured])),p95=float(np.percentile([r[k] for r in measured],95))) for k in keys}
 save(root/'report.json',dict(architecture=mode,gpu=torch.cuda.get_device_name(),load_seconds=load,records=records,summary=summary,frames_per_processing_second=len(measured)/sum(r['total_seconds'] for r in measured),warmup_frames=4,measured_frames=32,parameters=params,encoder_audit=runner.encoder.audit if mode=='teacher' else {'backbone_sha256':sha(ROOT/'bundle-seed0/backbone.pth'),'feature_rounding':'float16 then float32'},loading_scope='Fresh process/model instance; filesystem cache state not controlled',peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,video=str(path),video_sha256=sha(path),source_fps=reader.fps,sampling_hz=10,detector_policy=read(HERE/'protocol.json')['training_candidates'],future_lookahead_seconds=.4 if mode=='teacher' else 0,synchronization='CUDA synchronized at stage boundaries; instrumented serial processing, not paced live latency',teacher_preprocessing_scope='Normalization and crop transforms; full-scene resize/RoI pooling remain in encoding',time=time.time()))
if __name__=='__main__':main()
