"""Isolated cross-actor association pilot; original student assets are read-only."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'comp765-cseb-student-20261021'))
import common as original
from common import *
import evaluate as legacy
BASE_ROOT=original.ROOT
HERE=Path(__file__).resolve().parent
ROOT=BASE_ROOT.parent/HERE.name
SCRATCH=Path('/dev/shm')/HERE.name
SEEDS=[0,1,2]
PILOT_CODE=HERE.parent/'comp765-student-association-20261026'
PILOT_ROOT=BASE_ROOT.parent/PILOT_CODE.name
CONDITIONS=['baseline','short_single','short_frequency','full_single','full_frequency','full_pair','full_pair_frequency']
def progress(phase,**kw):
 d=dict(time=time.time(),phase=phase,**kw);save(ROOT/'progress.json',d);print(json.dumps(d),flush=True)
def checkpoint(seed):return BASE_ROOT/'runs'/f'distilled-seed{seed}'/'best.pt'
def signature():return dict(protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(ROOT/'freeze.json'))
def verify():
 for path,h in read(ROOT/'freeze.json')['files'].items():assert sha(path)==h,path
