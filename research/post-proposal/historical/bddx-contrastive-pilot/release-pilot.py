"""Release the held pilot only after the archive is verified and manifest validates."""
import json,subprocess,time,os
from pathlib import Path
root=Path('/work/bbyrd1/bddx-pilot-20260911');deadline=time.monotonic()+2400
while not (root/'data-ready.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Pilot remains held: verified data never became ready')
 time.sleep(10)
os.environ['PYTHONPATH']=str(root/'packages')
subprocess.run(['/work/bbyrd1/road_crop/py311/bin/python',str(root/'code/prepare.py'),'--videos',str(root/'data/BDDX_Processed/videos'),'--conversations',str(root/'data/bddx-conversation.json'),'--annotations',str(root/'data/BDD-X-Annotations_v1.csv'),'--train-split',str(root/'data/train.txt'),'--out',str(root/'manifest.json')],check=True)
subprocess.run(['scontrol','release','729087'],check=True)
(root/'launch.json').write_text(json.dumps({'smoke_job':729086,'pilot_job':729087,'dependency':'afterok:729086','data_verified':True,'state':'released; awaiting successful smoke and GPU scheduling'},indent=2)+'\n')
print('Pilot 729087 released; dependency on smoke 729086 retained.',flush=True)
