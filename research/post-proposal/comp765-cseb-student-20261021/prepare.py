"""Resumable training-only teacher targets; label-only selection features."""
from common import *
from PIL import Image
def main():
 setup();verify();splits=read(ROOT/'split-manifest.json');old=read(EXPANDED/'manifest.json');ann=read(p.ROOT/'road_waymo_trainval_v1.1.json');labels=old['labels'];dest=ROOT/'cache';dest.mkdir(exist_ok=True)
 records=read(ROOT/'data-manifest.json')['sequences'] if (ROOT/'data-manifest.json').exists() else []
 byvideo={r['video']:r for r in records};oldrows={r['video']:r for r in old['sequences']};started=time.perf_counter();completed=sum(r['frames'] for r in records)
 for r in records:assert sha(r['path'])==r['sha256']
 backbone=SharedBackbone(p.WEIGHTS).cuda();teacher=None;detector=None
 manifest=dict(signature=signature(),labels=labels,groups={k:splits['groups'][k] for k in ['train','selection']},sequences=records)
 for split in ['train','selection']:
  pending=[v for v in splits['groups'][split] if v not in byvideo]
  if not pending:continue
  if split=='train':teacher=p.CSEBInference(p.TEACHER,p.ENCODER,p.YOLO,ROOT/'yolo-settings');detector=teacher.detector
  else:
   if teacher is not None:del teacher;teacher=None;torch.cuda.empty_cache()
   if detector is None:
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'yolo-settings');from ultralytics import YOLO
    detector=YOLO(str(p.YOLO))
  for video in pending:
   oldrec=oldrows[video];assert sha(oldrec['path'])==oldrec['sha256'];prior=torch.load(oldrec['path'],map_location='cpu',weights_only=True);prior={r['fid']:r for r in prior}
   start=splits['start_frames'][video];seq=[];digests={};reused=0;teacher_frames=[]
   for fid in range(start,start+32):
    tick=time.perf_counter();frame=ann['db'][video]['frames'].get(str(fid),{});annotated=frame.get('annotated')==1
    if fid in prior:
     row=dict(prior[fid]);row['annotated']=True
     if split=='selection':row.pop('teacher')
     for f in range(fid-3,fid+5) if split=='train' else [fid]:
      path=image_path(video,f);digests[str(f)]=sha(path)
      assert digests[str(f)]==oldrec['frame_sha256'][str(f)]
     reused+=1
    else:
     image=np.array(Image.open(image_path(video,fid)).convert('RGB'));digests[str(fid)]=sha(image_path(video,fid))
     if not annotated:boxes=torch.empty((0,4));agents=torch.empty(0,dtype=torch.long);q=torch.empty(0);raw=None
     elif split=='train':
      imgs=[]
      for f in range(fid-3,fid+5):
       path=image_path(video,f);imgs.append(np.array(Image.open(path).convert('RGB')));digests[str(f)]=sha(path)
      out=teacher.predict_clip(np.stack(imgs),3);boxes=torch.from_numpy(out['boxes']);agents=torch.from_numpy(out['agents']);q=torch.from_numpy(out['confidence']);assert (q>0).all()
      raw=torch.from_numpy(out['scores']['blend'])/q[:,None];raw[:,0]=1;assert torch.isfinite(raw).all() and (raw>=0).all() and (raw<=1.00001).all();raw=raw.clamp(0,1);teacher_frames.append(fid)
     else:
      d=detector.predict(image[:,:,::-1].copy(),imgsz=1280,conf=.25,iou=.7,max_det=300,device=0,half=True,verbose=False)[0]
      boxes=d.boxes.xyxyn.float().cpu();agents=d.boxes.cls.long().cpu();q=d.boxes.conf.float().cpu();raw=None
     roi,scene=backbone(image,boxes)
     target=p.mapped_targets(frame,ann,labels,boxes.numpy(),agents.numpy()) if annotated else torch.empty((0,184))
     row=dict(fid=fid,roi=roi.half().cpu(),scene=scene.half().cpu(),boxes=boxes,agents=agents,q=q,labels=target,annotated=annotated)
     if split=='train':row['teacher']=raw if raw is not None else torch.empty((0,184))
    assert ('teacher' in row)==(split=='train')
    seq.append(row);completed+=1
    if fid%4==0:progress('preparation',split=split,video=video,fid=fid,frames=completed,total=2432,seconds=time.perf_counter()-tick,elapsed=time.perf_counter()-started)
   path=dest/f'{video}.pt';p.atomic_save(seq,path)
   rec=dict(video=video,split=split,path=str(path),sha256=sha(path),frames=32,rows=sum(len(r['boxes']) for r in seq),annotated_frames=sum(r['annotated'] for r in seq),reused_frames=reused,new_teacher_target_frames=teacher_frames,frame_sha256=digests)
   records.append(rec);byvideo[video]=rec;save(ROOT/'data-manifest.json',manifest)
 manifest['sequences']=[byvideo[v] for k in ['train','selection'] for v in splits['groups'][k]];save(ROOT/'data-manifest.json',manifest)
 save(ROOT/'preparation-complete.json',dict(signature=signature(),data_manifest_sha256=sha(ROOT/'data-manifest.json'),time=time.time(),seconds=time.perf_counter()-started,frames=2432,teacher_targets_only_train=True))
 progress('prepared',frames=2432,seconds=time.perf_counter()-started)
if __name__=='__main__':main()
