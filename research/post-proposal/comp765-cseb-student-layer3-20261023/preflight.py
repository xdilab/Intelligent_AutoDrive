"""Pre-results gradient, pairing, BN, causal, resume and throughput checks."""
from common import *
from training_model import *
from prepare import encode_sequence
from train import update,capture,restore,sequence
from PIL import Image
import copy,ast

def main():
 setup();rec=read(ORIGINAL_ROOT/'data-manifest.json')['sequences'][0];prefix=Prefix().cuda();start=time.perf_counter();x,d=encode_sequence(rec,prefix);torch.cuda.synchronize();prep=time.perf_counter()-start
 rows=[to_gpu(r) for r in torch.load(rec['path'],weights_only=True)];x=x.cuda();m=TrainModel(False).cuda()
 with torch.no_grad():pooled=features(m.stage,x,rows)
 seq=dict(prefix=x,rows=rows,pooled=pooled);checks={};timings={};initials={}
 for seed in [0,1,2]:
  hs=[]
  for tuned in [False,True]:
   setup(seed);m=TrainModel(tuned).cuda();hs.append(tensor_hash(m.head.state_dict()))
  assert hs[0]==hs[1]==read(ORIGINAL_ROOT/'runs'/f'supervised-seed{seed}'/'training-complete.json')['initialization_sha256'];initials[str(seed)]=hs[0]
 checks['all_arms_and_historical_head_initialization']=initials
 for tuned in [False,True]:
  setup(0);m=TrainModel(tuned).cuda();opt=optimizer(m);before={k:v.clone() for k,v in m.stage.state_dict().items()};loss=update(m,opt,seq,1.)
  for k,v in m.stage.state_dict().items():
   if not tuned or 'running_' in k or 'num_batches' in k:assert torch.equal(v,before[k]),k
  if tuned:assert not torch.equal(m.stage.state_dict()['0.conv1.weight'],before['0.conv1.weight']);assert m.stage[0].conv1.weight.grad.abs().sum()>0
  checks[str(tuned)+'_finite_smoke_loss']=loss
  ck=copy.deepcopy(capture(m,opt));p.atomic_save(ck,ROOT/'preflight-resume.pt');expected_loss=update(m,opt,seq,1.);expected={k:v.clone() for k,v in m.state_dict().items()};rng=[torch.rand(3).tolist(),torch.rand(3,device='cuda').tolist(),np.random.rand(3).tolist(),random.random()]
  n=TrainModel(tuned).cuda();no=optimizer(n);restore(n,no,torch.load(ROOT/'preflight-resume.pt',map_location='cpu',weights_only=False));actual_loss=update(n,no,seq,1.)
  maxdiff=max(float((v-expected[k]).abs().max()) for k,v in n.state_dict().items());assert abs(actual_loss-expected_loss)<1e-6
  for k,v in n.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=1e-5,atol=1e-6)
  assert rng==[torch.rand(3).tolist(),torch.rand(3,device='cuda').tolist(),np.random.rand(3).tolist(),random.random()];checks[str(tuned)+'_resume_max_abs']=maxdiff
  del n,no,ck,before,expected;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(8):update(m,opt,seq,1.)
  torch.cuda.synchronize();timings['tuned' if tuned else 'frozen']=dict(seconds_per_sequence=(time.perf_counter()-start)/8,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
  with torch.no_grad():
   z,_=sequence(m,seq);changed=dict(seq,prefix=seq['prefix'].clone(),pooled=[dict(r) for r in pooled]);changed['prefix'][-1]+=1;changed['pooled'][-1]['scene']=changed['pooled'][-1]['scene']+1;zz,_=sequence(m,changed);nrows=sum(len(r['boxes']) for r in rows[:-1]);torch.testing.assert_close(z[:nrows],zz[:nrows],rtol=0,atol=0);zr,_=sequence(m,seq);torch.testing.assert_close(z,zr,rtol=0,atol=0)
 checks['BN_running_stats_frozen_and_causal_prefix_and_reset']=True
 logits=torch.zeros((2,184),device='cuda',requires_grad=True);y=torch.zeros_like(logits);y[:,[11,12,98]]=1;objective(logits,y,y*.8,1).backward();assert (logits.grad[:,0]==0).all() and (logits.grad[:,[11,12,98]]<0).all();checks['multilabel_slot0_mask']=True
 q=torch.tensor([.25,.5],device='cuda');prob=torch.full((2,184),.6,device='cuda');out=p.detector_scores(prob,q);torch.testing.assert_close(out[:,1],prob[:,1]*q);torch.testing.assert_close(out[:,0],q);checks['confidence_once']=True
 def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
 a=funcs(ORIGINAL_CODE/'evaluate.py');b=funcs(HERE/'evaluate.py');assert all(a[k]==b[k] for k in ['load_official','one_class']);checks['official_metric_and_inputs_unchanged']=True
 # Verify split prefix -> stage equals original per-frame encoder; batch32 numerical differences recorded.
 original=SharedBackbone(WEIGHTS).cuda();fresh=TrainModel(False).cuda();image=np.array(Image.open(image_path(rec['video'],rows[0]['fid'])).convert('RGB'))
 with torch.no_grad():
  roi,scene=original(image,rows[0]['boxes']);r=features(fresh.stage,prefix(image),rows[:1])[0];torch.testing.assert_close(r['roi'],roi.half().float(),rtol=0,atol=0);torch.testing.assert_close(r['scene'],scene.half().float(),rtol=0,atol=0)
 checks['split_encoder_framewise_exact']=True;checks['initial_batch32_vs_frame1_roi_max_abs']=float((pooled[0]['roi']-r['roi']).abs().max());checks['initial_batch32_vs_frame1_scene_max_abs']=float((pooled[0]['scene']-r['scene']).abs().max())
 params=dict(semantic_total=sum(v.numel() for v in original.parameters())+sum(v.numel() for v in m.head.parameters()),head=sum(v.numel() for v in m.head.parameters()),frozen_encoder_total=sum(v.numel() for v in original.parameters()),layer3=sum(v.numel() for v in m.stage.parameters()))
 save(ROOT/'preflight.json',dict(passed=True,time=time.time(),checks=checks,parameters=params));save(ROOT/'benchmark.json',dict(time=time.time(),prepare32frames_seconds=prep,training=timings,gpu=torch.cuda.get_device_name(),prefix_cache_gib=x.numel()*x.element_size()*76/2**30,parameters=params,scope='Eight actual32frame updates each arm after warmup, cached prefix onGPU; no AP inspected'))
 (ROOT/'preflight-resume.pt').unlink();print(json.dumps(read(ROOT/'benchmark.json'),indent=2));progress('preflight-passed')
if __name__=='__main__':main()
