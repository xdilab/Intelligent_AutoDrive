from assoc_common import *
import subprocess,platform

def main():
 if (ROOT/'freeze.json').exists():verify();print('Existing freeze verified');return
 assert read(ROOT/'preflight.json')['passed'];assert read(ROOT/'protocol-before-mining.json')['sha256']==sha(HERE/'protocol.json');assert read(ROOT/'asset-audit.json')['passed']
 names=['assoc_common.py','mine.py','correction.py','single_correction.py','preflight.py','official.py','run_video.py','runtime.py','replay_audit.py','protocol.json'];paths=[HERE/n for n in names]+[ROOT/n for n in ['rules.json','short-rules.json','single-rules.csv','pair-candidates.csv','split-manifest.json']]+[BASE_ROOT/'data-manifest.json',BASE_ROOT/'official-assets-audit.json',PILOT_ROOT/'score-diagnostics.json',OLD/'model.py',OLD/'pilot.py',PARENT/'demo/predict.py',PARENT/'comp765-cseb-student-20261021/common.py',PARENT/'comp765-cseb-student-20261021/evaluate.py',PARENT/'stage56-full/metric.py',PARENT/'stage56-full/baseline-core.py',INPUTS['frames'],INPUTS['counts']]+[checkpoint(s) for s in SEEDS]
 save(ROOT/'freeze.json',dict(time=time.time(),before_new_official_AP=True,files={str(x):sha(x) for x in paths}));save(ROOT/'provenance.json',dict(time=time.time(),base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),source_checkpoints=read(ROOT/'preflight.json')['checkpoints'],assets=read(ROOT/'asset-audit.json'),scope='No retraining. Same3original distilled students. Training-only full-video mining and residual two-event rule extension. Strength fixed1; no new AP tuning.'))
 print('Frozen',len(paths),'files')
if __name__=='__main__':main()
