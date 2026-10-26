"""Freeze a fresh reproduction after development selection; never overwrite a freeze."""
from assoc_common import *
import subprocess,platform

def main():
 if (ROOT/'freeze.json').exists():verify();print('Existing freeze verified; not overwritten');return
 assert read(ROOT/'preflight.json')['passed'];assert read(ROOT/'protocol-before-mining.json')['sha256']==sha(HERE/'protocol.json')
 paths=list(HERE.glob('*.py'))+[HERE/'protocol.json',ROOT/'rules.json',ROOT/'rules.csv',ROOT/'development-selection.json',ROOT/'split-manifest.json',BASE_ROOT/'data-manifest.json',BASE_ROOT/'official-assets-audit.json',OLD/'model.py',OLD/'pilot.py',PARENT/'demo/predict.py',PARENT/'comp765-cseb-student-20261021/common.py',PARENT/'comp765-cseb-student-20261021/evaluate.py',PARENT/'stage56-full/metric.py',PARENT/'stage56-full/baseline-core.py',INPUTS['frames'],INPUTS['counts']]+[checkpoint(s) for s in SEEDS]
 save(ROOT/'freeze.json',dict(time=time.time(),before_official_AP=True,files={str(x):sha(x) for x in paths}))
 save(ROOT/'provenance.json',dict(time=time.time(),base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),original_provenance=str(BASE_ROOT/'provenance.json'),original_provenance_sha256=sha(BASE_ROOT/'provenance.json'),source_checkpoints=read(ROOT/'preflight.json')['checkpoints'],development_selection=read(ROOT/'development-selection.json')['strengths'],scope='No student retraining. Original seeds reused for fixed postprocessing comparison.'))
 print('Frozen',len(paths),'files')
if __name__=='__main__':main()
