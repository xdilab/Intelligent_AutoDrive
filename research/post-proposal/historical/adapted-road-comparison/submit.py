"""Submit one held preflight followed by an afterok-only comparison chain."""
import json,subprocess
from pathlib import Path
root=Path('/work/bbyrd1/adapted-road-20260911');old=Path('/work/bbyrd1/proposal-study-20260910/data')
for p in ['rows','cache','smoke','results/metrics/original','results/metrics/adapted','runs/original','runs/adapted','data/original','data/adapted']:(root/p).mkdir(parents=True,exist_ok=True)
for condition in ['original','adapted']:
 for name in ['flat_alphas.pt','dets_v8x_best_val_fullcand.pkl','dets_i3d_val_fullcand.pkl']:
  dest=root/'data'/condition/name
  if not dest.exists():dest.symlink_to(old/name)
assert not (root/'results/jobs.json').exists(),'Chain already submitted'
ids={};previous=None
for name in ['smoke','cache','merge','train','eval']:
 cmd=['sbatch','--parsable']
 if previous:cmd.append('--dependency=afterok:'+previous)
 else:cmd.append('--hold')
 cmd.append(str(root/'code'/(name+'.sbatch')))
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0];ids[name]=job;previous=job
 (root/'results/jobs.json').write_text(json.dumps(ids,indent=2)+'\n')
 print(name,job,flush=True)
