"""Read-only historical audit and training-input-only throughput benchmark."""
from common import *
from PIL import Image
import platform,subprocess
def main():
 ROOT.mkdir(exist_ok=True,parents=True);setup()
 a=read(p.ROOT/'road_waymo_trainval_v1.1.json');old=read(EXPANDED/'manifest.json');prep=read(p.PREP);official=read(INPUTS['frames'])
 splits={k:old['groups'][k] for k in ['train','selection']};splits['final']=sorted({x.rsplit('_',1)[0] for x in official['frames']})
 starts={}
 for split in ['train','selection']:
  for v in splits[split]:
   starts[v]=old['start_frames'][v]
   assert all(image_path(v,f).is_file() for f in range(starts[v]-3,starts[v]+36))
 teacher=torch.load(p.TEACHER,map_location='cpu',weights_only=False);gate=teacher['signature']['selection_videos']
 assert not set(splits['final'])&set(prep['videos']['train']+prep['videos']['dev']+gate)
 assert len(set(sum(splits.values(),[])))==274
 assert read(INPUTS['teacher_result'])['signature']['checkpoint_sha256']==sha(p.TEACHER)
 assert all(official[k]==read(INPUTS['teacher_result'])[k] for k in ['candidate_sha256','frame_sha256'])
 labels=old['labels'];assert all(labels[h]==a[h+'_labels']==official['labels'][h] for h in HEADS[1:])
 continuous={v:max(int(x.rsplit('_',1)[1]) for x in official['frames'] if x.startswith(v+'_')) for v in splits['final']}
 assert all(image_path(v,f).is_file() for v,n in continuous.items() for f in range(1,n+1))
 save(ROOT/'split-manifest.json',dict(groups=splits,start_frames=starts,sequence_length=32,final_last_frame=continuous,teacher_training=prep['videos']['train'],teacher_selection=prep['videos']['dev'],teacher_blend_selection=gate,overlap_final_training_selection=[],historical_student_holdout=old['groups']['heldout'],final_prior_use='Official validation repeatedly used in earlier thesis research; not a pristine unseen test set.',selection_prior_use='Student selection is a subset of teacher historical development.',source_expanded_manifest_sha256=sha(EXPANDED/'manifest.json')))
 audit={'time':time.time(),'expanded_status':read(EXPANDED/'job.json'),'expanded_results':read(EXPANDED/'results.json'),'teacher':str(p.TEACHER),'teacher_sha256':sha(p.TEACHER),'teacher_signature':teacher['signature'],'teacher_result':str(INPUTS['teacher_result']),'teacher_result_sha256':sha(INPUTS['teacher_result']),'unannotated_frames':{v:[f for f in range(starts[v],starts[v]+32) if a['db'][v]['frames'].get(str(f),{}).get('annotated')!=1] for v in starts},'shifted_starts':{v:[old['start_frames'][v],s] for v,s in starts.items() if s!=old['start_frames'][v]},'final_scored_frames':len(official['frames']),'final_processed_frames':sum(continuous.values()),'class_groups':groups(labels['triplet']),'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}
 save(ROOT/'audit.json',audit);del a,teacher
 # Four training videos, four frames each, excluded first frame per video as warmup.
 settings=ROOT/'yolo-settings';settings.mkdir(exist_ok=True)
 start=time.perf_counter();teacher=p.CSEBInference(p.TEACHER,p.ENCODER,p.YOLO,settings);backbone=SharedBackbone(p.WEIGHTS).cuda();model=Student().cuda();loaded=time.perf_counter()-start
 rows=[];records=[]
 for video in splits['train'][:4]:
  for j in range(4):
   fid=starts[video]+8+j;t=time.perf_counter()
   imgs=np.stack([np.array(Image.open(image_path(video,f)).convert('RGB')) for f in range(fid-3,fid+5)])
   decoded=time.perf_counter();out=teacher.predict_clip(imgs,3);boxes=torch.from_numpy(out['boxes']);agents=torch.from_numpy(out['agents']);q=torch.from_numpy(out['confidence']);roi,scene=backbone(imgs[3],boxes)
   torch.cuda.synchronize();end=time.perf_counter();records.append(dict(video=video,fid=fid,proposals=len(boxes),seconds=end-t,decode_seconds=decoded-t,warmup=j==0,**out['timings']))
   rows.append(dict(roi=roi.half().float(),scene=scene.half().float(),boxes=boxes.cuda(),agents=agents.cuda(),teacher=torch.from_numpy(out['scores']['blend']).cuda()/q.cuda()[:,None],labels=torch.zeros((len(q),184),device='cuda')))
   progress('preflight-benchmark',**records[-1])
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);times=[]
 for j in range(6):
  torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);state=None;zs=[];ys=[];ts=[]
  for row in (rows*2):
   z,state=model(row['roi'],row['scene'],row['boxes'],row['agents'],state);zs.append(z);ys.append(row['labels']);ts.append(row['teacher'])
  loss=objective(torch.cat(zs),torch.cat(ys),torch.cat(ts),1);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-t)
 result=dict(load_seconds=loaded,records=records,mean_preparation_seconds=float(np.mean([r['seconds'] for r in records if not r['warmup']])),training_32frame_step_seconds=times,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,scope='Training-only inputs; synthetic zero labels for throughput, not a scientific training run;32-frame BPTT,64 sequences/epoch,30epochs,6runs.')
 save(ROOT/'benchmark.json',result);progress('benchmark-complete',**{k:v for k,v in result.items() if k!='records'})
if __name__=='__main__':main()
