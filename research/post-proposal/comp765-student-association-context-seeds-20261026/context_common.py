"""Ten-seed wrapper; numerical inference/evaluation imported unchanged from frozen parent."""
import sys,importlib.util
from pathlib import Path
CONTEXT_CODE=Path(__file__).resolve().parent.parent/'comp765-student-association-context-20261026'
sys.path.insert(0,str(CONTEXT_CODE))
import assoc_common as context
from assoc_common import *
HERE=Path(__file__).resolve().parent
CONTEXT_CODE=HERE.parent/'comp765-student-association-context-20261026'
CONTEXT_ROOT=BASE_ROOT.parent/CONTEXT_CODE.name
EXPANSION_ROOT=BASE_ROOT.parent/'comp765-cseb-student-seeds-20261022'
ROOT=BASE_ROOT.parent/HERE.name
SCRATCH=Path('/dev/shm')/HERE.name
SEEDS=list(range(10))
BATCHES=[[3,4,5,6],[7,8,9]]
def checkpoint(seed):return (BASE_ROOT if seed<3 else EXPANSION_ROOT)/'runs'/f'distilled-seed{seed}'/'best.pt'
def backup(seed):return (BASE_ROOT if seed<3 else EXPANSION_ROOT)/'verified-checkpoint-backup'/f'distilled-seed{seed}'/'best.pt'
def progress(phase,**kw):
 d=dict(time=time.time(),phase=phase,**kw);save(ROOT/'progress.json',d);print(json.dumps(d),flush=True)
def signature():return dict(protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(ROOT/'freeze.json'))
def verify():
 for path,h in read(ROOT/'freeze.json')['files'].items():assert sha(path)==h,path
def batch_name(seeds):return 'seeds'+str(seeds[0])+'-'+str(seeds[-1])
def evaluator(seeds):
 spec=importlib.util.spec_from_file_location('frozen_context_evaluator',CONTEXT_CODE/'official.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);name=batch_name(seeds)
 for key,value in dict(ROOT=ROOT/'batches'/name,SCRATCH=SCRATCH/name,SEEDS=seeds,checkpoint=checkpoint,signature=signature,verify=verify,progress=lambda phase,**kw:progress(phase,batch=name,**kw)).items():setattr(m,key,value)
 return m
