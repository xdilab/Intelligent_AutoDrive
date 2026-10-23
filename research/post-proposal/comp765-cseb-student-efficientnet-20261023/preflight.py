"""Prelaunch numerical/causal/resume checks and representative throughput."""
from common import *
from prepare import encode_sequence
from train import update,capture,restore,sequence
from PIL import Image
import copy,ast

def main():
 setup();rec=read(ORIGINAL_ROOT/'data-manifest.json')['sequences'][0];b=SharedBackbone(WEIGHTS).cuda();checks={};image=np.array(Image.open(image_path(rec['video'],read(ROOT/'split-manifest.json')['start_frames'][rec['video']])).convert('RGB'))
 with torch.inference_mode():
  roi,scene=b(image,np.array([[0.,0.,1.,1.]],np.float32));assert roi.shape==(1,1280) and scene.shape==(1280,);empty,_=b(image,np.empty((0,4),np.float32));assert empty.shape==(0,1280)
  x=torch.randn(1,3,448,672,device='cuda');f=b.body(x);assert f.shape==(1,1280,14,21)
  # Actual feature-grid dimensions drive coordinates; identical tensors reproduce the backbone path.
  from torch.nn import functional as F
  from torchvision.ops import roi_align
  x=torch.as_tensor(image,device='cuda').permute(2,0,1)[None].float()/255;x=F.interpolate(x,(448,672),mode='bilinear',align_corners=False);mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
  with torch.autocast('cuda',dtype=torch.bfloat16):f=b.body((x-mean)/std)
  f=f.float();manual=roi_align(f,torch.tensor([[0.,0.,0.,21.,14.]],device='cuda'),output_size=3,aligned=True).mean((-1,-2));torch.testing.assert_close(roi,manual,rtol=0,atol=0)
 checks['feature_grid_and_roi_alignment']=True;torch.cuda.synchronize();t=time.perf_counter();raw,digests=encode_sequence(rec,b);torch.cuda.synchronize();prep_seconds=time.perf_counter()-t
 seq=[to_gpu(v) for v in raw];initials={};losses=[]
 for seed in [0,1,2]:
  pair=[]
  for kd in [0.,1.]:
   setup(seed);m=Student().cuda();pair.append(tensor_hash(m.state_dict()));o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);losses.append(update(m,o,seq,kd))
  assert pair[0]==pair[1];initials[str(seed)]=pair[0]
 checks['paired_initialization']=initials;checks['smoke_losses']=losses
 ck=copy.deepcopy(capture(m,o));p.atomic_save(ck,ROOT/'preflight-resume.pt');expected_loss=update(m,o,seq,1);expected=tensor_hash(m.state_dict());expected_rng=[torch.rand(3).tolist(),torch.rand(3,device='cuda').tolist(),np.random.rand(3).tolist(),random.random()]
 n=Student().cuda();no=torch.optim.AdamW(n.parameters(),lr=.001,weight_decay=.01);restore(n,no,torch.load(ROOT/'preflight-resume.pt',weights_only=False,map_location='cpu'));actual_loss=update(n,no,seq,1);assert expected_loss==actual_loss and tensor_hash(n.state_dict())==expected;assert expected_rng==[torch.rand(3).tolist(),torch.rand(3,device='cuda').tolist(),np.random.rand(3).tolist(),random.random()];checks['exact_optimizer_rng_resume']=True
 with torch.no_grad():
  z,_=sequence(m,seq);changed=[dict(v) for v in seq];changed[-1]=dict(changed[-1],scene=changed[-1]['scene']+20);zz,_=sequence(m,changed);prefix=sum(len(r['boxes']) for r in seq[:-1]);torch.testing.assert_close(z[:prefix],zz[:prefix],rtol=0,atol=0);reset,_=sequence(m,seq);torch.testing.assert_close(z,reset,rtol=0,atol=0)
 checks['causal_prefix_and_video_reset']=True
 logits=torch.zeros((2,184),device='cuda',requires_grad=True);y=torch.zeros_like(logits);teacher=y.clone();y[:,[11,12,98]]=1;teacher[:,[11,12,98]]=.8;objective(logits,y,teacher,1).backward();assert (logits.grad[:,0]==0).all() and (logits.grad[:,[11,12,98]]<0).all();checks['multilabel_slot0_mask']=True
 prob=torch.full((2,184),.6,device='cuda');q=torch.tensor([.25,.5],device='cuda');out=p.detector_scores(prob,q);torch.testing.assert_close(out[:,1],prob[:,1]*q);torch.testing.assert_close(out[:,0],q);checks['confidence_once']=True
 def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 a=funcs(ORIGINAL_CODE/'train.py');c=funcs(HERE/'train.py');assert all(a[k]==c[k] for k in ['sequence','update','selection','capture','restore','load_data']);a=funcs(ORIGINAL_CODE/'evaluate.py');c=funcs(HERE/'evaluate.py');assert all(a[k]==c[k] for k in ['load_official','step','one_class']);checks['training_and_scoring_functions_unchanged']=True
 # Six independent heads on one representative32-frame sequence, no AP inspection.
 from evaluate import step
 models={str(i):Student().cuda().eval() for i in range(6)};states={k:None for k in models};torch.cuda.synchronize();t=time.perf_counter()
 for r in raw:
  im=np.array(Image.open(image_path(rec['video'],r['fid'])).convert('RGB'));step(models,states,b,im,r['boxes'].cuda(),r['agents'].cuda(),r['q'].cuda())
 torch.cuda.synchronize();inference_seconds=time.perf_counter()-t
 setup(0);m=Student().cuda();o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);t=time.perf_counter()
 for _ in range(16):update(m,o,seq,1)
 torch.cuda.synchronize();train_seconds=time.perf_counter()-t
 params=dict(frozen_backbone=sum(v.numel() for v in b.parameters()),trainable_head=sum(v.numel() for v in m.parameters()));params['semantic_total']=sum(params.values())
 save(ROOT/'preflight.json',dict(passed=True,time=time.time(),checks=checks,parameters=params));save(ROOT/'benchmark.json',dict(time=time.time(),prepare32frames_seconds=prep_seconds,train16sequences_seconds=train_seconds,six_head_inference32frames_seconds=inference_seconds,gpu=torch.cuda.get_device_name(),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,parameters=params,scope='One actual32frame training sequence; six random heads with no AP inspection. Broader full-video throughput measured before full evaluator launch.'))
 print(json.dumps(read(ROOT/'benchmark.json'),indent=2));progress('preflight-passed')
if __name__=='__main__':main()
