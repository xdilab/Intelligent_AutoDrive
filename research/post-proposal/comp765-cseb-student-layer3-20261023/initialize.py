"""Initialize a new isolated root; freeze only after successful preflight.

For an independent rerun, set ROOT in common.py to a new directory first.
Existing freezes are verified, never overwritten. Original assets are read-only.
"""
from common import *
import argparse,shutil,importlib.metadata

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
 for name in ['split-manifest.json','data-audit.json','official-assets-audit.json','annotation-mapping-audit.json','detector-history-audit.json','teacher-numerical-parity-audit.json']:
  src=ORIGINAL_ROOT/name;dest=ROOT/name
  if dest.exists():assert sha(src)==sha(dest)
  else:shutil.copy2(src,dest)
 if not a.freeze:print('Initialized',ROOT);return
 if (ROOT/'freeze.json').exists():verify();print('Existing freeze verified, unchanged');return
 assert read(ROOT/'preflight.json')['passed'];assert read(ROOT/'preflight.json')['parameters']==read(HERE/'protocol.json')['student_parameters']
 paths=[HERE/n for n in ['model.py','training_model.py','common.py','prepare.py','train.py','evaluate.py','preflight.py','evaluation_preflight.py','standalone.py','protocol.json','PROVENANCE-ADDENDUM.md']]+[ROOT/'split-manifest.json',ORIGINAL_ROOT/'data-manifest.json',WEIGHTS,OLD/'pilot.py',PARENT/'stage56-full/metric.py',PARENT/'stage56-full/baseline-core.py',INPUTS['frames'],INPUTS['counts'],p.YOLO,p.TEACHER]
 paths += [ORIGINAL_ROOT/'runs'/f'{c}-seed{s}'/'detector-results.json' for s in [0,1,2] for c in ['supervised','distilled']]
 save(ROOT/'freeze.json',dict(time=time.time(),scope='Frozen before new full training/evaluation; numeric model, loss, preprocessing, evaluator, protocol, data and baseline results',files={str(path):sha(path) for path in paths}))
 versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','opencv-python','ultralytics','Pillow','matplotlib']};save(ROOT/'environment.json',dict(python=sys.version,executable=sys.executable,versions=versions,PYTHONPATH=os.environ.get('PYTHONPATH')));print('Frozen',sha(ROOT/'freeze.json'))
if __name__=='__main__':main()
