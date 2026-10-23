from common import *
from training_model import TrainModel,Prefix
from evaluate import step
from PIL import Image
setup();rec=read(ORIGINAL_ROOT/'data-manifest.json')['sequences'][0];rows=torch.load(rec['path'],weights_only=True);models={f'{c}-seed{s}':TrainModel(c.startswith('tuned')).cuda().eval() for s in [0,1,2] for c in CONDITIONS};b=Prefix().cuda();states={k:None for k in models};torch.cuda.synchronize();t=time.perf_counter()
for row in rows:
 image=np.array(Image.open(image_path(rec['video'],row['fid'])).convert('RGB'));step(models,states,b,image,row['boxes'].cuda(),row['agents'].cuda(),row['q'].cuda())
torch.cuda.synchronize();seconds=time.perf_counter()-t;save(ROOT/'inference-smoke-benchmark.json',dict(frames=32,seconds=seconds,frames_per_second=32/seconds,heads=12,layer3_branches=7,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,scope='Actual32trainingframes, training candidates, twelve random heads; no AP. Official full-candidate benchmark after training.',time=time.time()));print(json.dumps(read(ROOT/'inference-smoke-benchmark.json'),indent=2))
