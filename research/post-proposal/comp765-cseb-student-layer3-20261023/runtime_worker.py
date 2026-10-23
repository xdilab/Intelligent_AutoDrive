"""Fresh-process raw-video timing for B0 and the preserved ResNet student."""
from common import *
from standalone import Runner
from infer_video import VideoReader
from torch.nn import functional as F
from torchvision.ops import roi_align

def sync():torch.cuda.synchronize();return time.perf_counter()
@torch.inference_mode()
def timed_student(runner,image):
 t=sync();d=runner.detector.predict(image[:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0];box=d.boxes.xyxyn.float();agent=d.boxes.cls.long();q=d.boxes.conf.float();dtime=sync()
 x=torch.as_tensor(image,device='cuda').permute(2,0,1)[None].float()/255;x=F.interpolate(x,size=(448,672),mode='bilinear',align_corners=False);mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None];x=(x-mean)/std;ptime=sync()
 with torch.autocast('cuda',dtype=torch.bfloat16):fmap=runner.backbone.body(x)
 fmap=fmap.float();h,w=fmap.shape[-2:];rois=torch.cat((box.new_zeros((len(box),1)),box*box.new_tensor([w,h,w,h])),1);roi=roi_align(fmap,rois,output_size=3,aligned=True).mean((-1,-2));scene=fmap.mean((-1,-2))[0];etime=sync()
 z,runner.state=runner.model(roi.half().float(),scene.half().float(),box,agent,runner.state);scores=z.sigmoid()*q[:,None];scores[:,0]=q;htime=sync();assert torch.isfinite(scores).all()
 out=dict(boxes=box.cpu().numpy(),agents=agent.cpu().numpy(),confidence=q.cpu().numpy(),scores=scores.cpu().numpy());end=sync()
 return out,dict(detector_seconds=dtime-t,preprocessing_seconds=ptime-dtime,encoding_seconds=etime-ptime,head_seconds=htime-etime,cpu_export_seconds=end-htime)

def main():
 mode=sys.argv[1];assert mode in ['tuned','frozen'];setup();root=ROOT/'runtime'/mode;root.mkdir(parents=True,exist_ok=True);bundle=ROOT/('bundle-seed0' if mode=='tuned' else 'bundle-frozen-seed0');torch.cuda.reset_peak_memory_stats();t=sync();runner=Runner(bundle,root/'yolo-settings');load=sync()-t
 params=dict(encoder=sum(v.numel() for v in runner.backbone.parameters()),heads=sum(v.numel() for v in runner.model.parameters()),detector=sum(v.numel() for v in runner.detector.model.parameters()));video=read(ROOT/'split-manifest.json')['groups']['train'][0];path=p.ROOT/'videos'/f'{video}.mp4';reader=VideoReader(path);records=[]
 with (root/'predictions.jsonl').open('w') as output:
  for fid in range(1,37):
   t=sync();image=reader.frame(fid);decoded=sync();out,times=timed_student(runner,image);serialized=time.perf_counter();output.write(json.dumps(dict(fid=fid,**{k:v.tolist() for k,v in out.items()}),separators=(',',':'))+'\n');output.flush();end=sync();candidate=hashlib.sha256(out['boxes'].tobytes()+out['agents'].tobytes()+out['confidence'].tobytes()).hexdigest()
   rec=dict(fid=fid,warmup=fid<=4,candidates=len(out['boxes']),candidate_sha256=candidate,input_fids=[fid],resolution=list(image.shape[:2]),decode_seconds=decoded-t,**times,json_export_seconds=end-serialized,total_seconds=end-t,gpu_allocated=torch.cuda.memory_allocated(),gpu_reserved=torch.cuda.memory_reserved());records.append(rec);progress('runtime-benchmark',architecture=mode,**rec)
 reader.cap.release();measured=[r for r in records if not r['warmup']];keys=[k for k in measured[0] if k.endswith('_seconds')];summary={k:dict(mean=float(np.mean([r[k] for r in measured])),median=float(np.median([r[k] for r in measured])),p95=float(np.percentile([r[k] for r in measured],95))) for k in keys}
 save(root/'report.json',dict(architecture=mode,gpu=torch.cuda.get_device_name(),load_seconds=load,records=records,summary=summary,frames_per_processing_second=len(measured)/sum(r['total_seconds'] for r in measured),warmup_frames=4,measured_frames=32,parameters=params,bundle_sha256=sha(bundle/'bundle.json'),loading_scope='Fresh process/model; warm filesystem cache not controlled',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,video=str(path),video_sha256=sha(path),source_fps=reader.fps,sampling_hz=10,detector_policy=read(HERE/'protocol.json')['training_candidates'],future_lookahead_seconds=0,synchronization='CUDA synchronized at every stage; serial file processing not sensor-to-decision latency',time=time.time()))
if __name__=='__main__':main()
