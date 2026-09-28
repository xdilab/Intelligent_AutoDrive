from pathlib import Path
import sys,json,hashlib
ROOT=Path('/data/repos/wiki/artifacts/contextual-gate-20260919');BASE=ROOT.parent/'contextual-roi-20260917';CODE=Path(__file__).parent
sys.path.insert(0,str(CODE.parent/'contextual-roi-all184'))
from train_cached import Cache,atomic_json,atomic_torch,file_sha,ap_values
from model import ContextualRoIHead

def parents(seed):return [BASE/'runs'/f'attention-classification-seed{seed}',BASE.parent/'contextual-roi-all184-20260918/runs'/f'attention-contrastive-all184-seed{seed}']
def signature(seed):
 return {'protocol':file_sha(ROOT/'protocol.json'),'preparation':file_sha(ROOT/'preparation.json'),'cache':file_sha(ROOT/'cache-ready.json'),'seed':seed,'parents':[file_sha(p/'best.pt') for p in parents(seed)],'expert_model':file_sha(CODE.parent/'contextual-roi-all184/model.py'),'code':{name:file_sha(CODE/name) for name in ['common.py','router.py','train_gate.py','predict_gate.py']}}
