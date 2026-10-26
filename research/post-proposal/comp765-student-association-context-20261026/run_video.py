"""Teacher-free adapter around an unchanged standalone student bundle."""
import argparse,gzip,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from correction import Correction

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--rules',type=Path,required=True);ap.add_argument('--strength',type=float,required=True);ap.add_argument('--video',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=0);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 opened=[]
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   path=Path(os.fsdecode(args[0])).resolve()
   if path.suffix in ['.pt','.pth','.safetensors','.npy','.npz','.pkl']:
    if path.parent!=a.bundle.resolve():raise RuntimeError('External model/cache asset forbidden: '+str(path))
    opened.append(str(path))
 sys.addaudithook(audit)
 spec=importlib.util.spec_from_file_location('unchanged_bundle_runner',a.bundle/'run.py');portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)
 begin=time.perf_counter();runner=portable.Runner(a.bundle,a.output/'yolo-settings');rules=json.loads(a.rules.read_text());assert rules['events']==runner.labels['duplex']+runner.labels['triplet'];correct=Correction(rules,'cuda');torch.cuda.synchronize();load=time.perf_counter()-begin
 cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert cap.isOpened() and fps>0 and count>0
 samples=math.floor((count-1)*10/fps)+1;limit=min(samples,a.frames) if a.frames else samples;next_index=0;records=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats();model_sha=hashlib.sha256()
 for k,v in sorted(runner.model.state_dict().items()):model_sha.update(k.encode());model_sha.update(v.detach().cpu().contiguous().numpy().tobytes())
 with torch.inference_mode(),gzip.open(a.output/'predictions.jsonl.gz','wt') as out:
  for fid in range(1,limit+1):
   total=time.perf_counter();t=total;index=min(count-1,round((fid-1)*fps/10))
   while next_index<=index:ok,image=cap.read();assert ok;next_index+=1
   assert next_index-1==index;rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);decode=time.perf_counter()-t;t=time.perf_counter()
   det=runner.detector.predict(rgb[:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0];box=det.boxes.xyxyn.float();agents=det.boxes.cls.long();q=det.boxes.conf.float();torch.cuda.synchronize();detector=time.perf_counter()-t;t=time.perf_counter()
   roi,scene=runner.backbone(rgb,box);z,runner.state=runner.model(roi.half().float(),scene.half().float(),box,agents,runner.state);torch.cuda.synchronize();student=time.perf_counter()-t;t=time.perf_counter()
   delta=correct.delta(z,box,agents,q,'full_pair');scores=correct.scores(z,q,delta,a.strength);torch.cuda.synchronize();correction=time.perf_counter()-t;t=time.perf_counter()
   assert torch.isfinite(scores).all() and torch.isfinite(runner.state).all();assert torch.equal(scores[:,:49],correct.scores(z,q)[:,:49]);pred=dict(fid=fid,source_index=index,boxes=box.cpu().tolist(),agents=agents.cpu().tolist(),confidence=q.cpu().tolist(),scores=scores.cpu().tolist());out.write(json.dumps(pred,separators=(',',':'))+'\n');export=time.perf_counter()-t
   records.append(dict(fid=fid,source_index=index,candidates=len(box),decode=decode,detector=detector,student=student,correction=correction,export=export,total=time.perf_counter()-total,gpu_allocated=torch.cuda.memory_allocated(),cpu_rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')))
 cap.release();summary={k:dict(median_ms=1000*float(np.median([r[k] for r in records[10:]])),p95_ms=1000*float(np.percentile([r[k] for r in records[10:]],95))) for k in ['decode','detector','student','correction','export','total']}
 report=dict(time=time.time(),frames=limit,source_video=str(a.video),source_sha256=portable.sha(a.video),source_resolution=list(rgb.shape),fps=fps,sampling_hz=10,load_seconds=load,processing_seconds=time.perf_counter()-start,records=records,summary=summary,warmup_excluded_frames=10,causal_source_indices_verified=True,external_assets_blocked=True,opened_model_assets=sorted(set(opened)),peak_gpu_bytes=torch.cuda.max_memory_allocated(),bundle_sha256=portable.sha(a.bundle/'bundle.json'),rules_sha256=portable.sha(a.rules),strength=a.strength,model_tensor_sha256=model_sha.hexdigest(),semantic_parameters=sum(v.numel() for v in runner.model.parameters())+sum(v.numel() for v in runner.backbone.parameters()),runtime_code_sha256=portable.sha(Path(__file__)),correction_code_sha256=portable.sha(Path(__file__).with_name('correction.py')),scope='Serial MP4 decode, detector, student, correction and compressed JSON export; excludes camera/display/transport; file throughput is not online latency')
 (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['records']}),flush=True)
if __name__=='__main__':main()
