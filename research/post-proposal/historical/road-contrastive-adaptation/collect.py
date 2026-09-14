import datetime,json,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent;W=A.parent.parent;D=A/'collected';D.mkdir(exist_ok=True);remote='/work/bbyrd1/road-contrastive-pilot-20260914'
subprocess.run(['rsync','-rt','ncshare:'+remote+'/results/',str(D)+'/'],check=True)
if (D/'preparation.json').exists() and not (A/'manifest.json').exists():subprocess.run(['rsync','-t','ncshare:'+remote+'/manifest.json',str(A/'manifest.json')],check=True)
jobs=[p.read_text().strip() for p in D.glob('*-job.txt')];status=subprocess.check_output(['ssh','ncshare','sacct -j '+','.join(jobs)+' -X --format=JobID,State,Elapsed,ExitCode -n'],text=True);now=datetime.datetime.now(datetime.timezone.utc).isoformat();(D/'status.txt').write_text(now+'\n'+status);print(status)
lines=['<!-- ROAD CONTRASTIVE PILOT STATUS START -->','',f'Last collected: **{now}**.','','```text',status.strip(),'```']
if (D/'preparation.json').exists():
 d=json.loads((D/'preparation.json').read_text());lines+=['','Audited crop manifest:']
 for split,c in d['counts'].items():lines.append(f"- {split}: {c['videos']} videos, {c['keyframes']} keyframes, {c['crop_rows']} crop rows, {c['triplet_classes_present']}/86 triplets represented.")
if (D/'smoke.json').exists():
 d=json.loads((D/'smoke.json').read_text());assert d['passed'] and d['frozen_sha256_before']==d['frozen_sha256_after'];lines+=['',f"Gradient smoke passed. Classification visual-gradient norm {d['classification_visual_gradient_norm']:.6g}; contrastive visual-gradient norm {d['contrastive_visual_gradient_norm']:.6g}; actual visual update norm {d['vision_parameter_update_l2']:.6g}; peak allocated GPU memory {d['peak_gpu_gb']:.2f} GB. Frozen parameters unchanged."]
reports=[]
for name in ['frozen','classification','contrastive']:
 p=D/f'{name}.json'
 if p.exists():
  d=json.loads(p.read_text());assert d['passed'] and d['frozen_sha256_before']==d['frozen_sha256_after'];reports.append(d)
if reports:
 lines+=['','Pilot internal-development crop AP only; one seed, not full detector results.','','| Condition | Baseline AP | Best baseline/epoch AP | Selected | Peak GPU GB |','|---|---:|---:|---|---:|']
 for d in reports:lines.append(f"| {d['condition']} | {d['baseline']['triplet_crop_AP']:.4f} | {max([d['baseline']['triplet_crop_AP']]+[e['triplet_crop_AP'] for e in d['epochs']]):.4f} | {Path(d['selected']).name} | {d['peak_gpu_gb']:.2f} |")
 lines+=['',f'{len(reports)}/3 pilot conditions complete.']
lines+=['','[Collected evidence](../artifacts/road-contrastive-adaptation/collected/).','','<!-- ROAD CONTRASTIVE PILOT STATUS END -->'];p=W/'directions/class-conditioned-language-fusion.md';s=p.read_text();start=s.index('<!-- ROAD CONTRASTIVE PILOT STATUS START -->');end=s.index('<!-- ROAD CONTRASTIVE PILOT STATUS END -->',start)+len('<!-- ROAD CONTRASTIVE PILOT STATUS END -->');p.write_text(s[:start]+'\n'.join(lines)+s[end:]);print('\n'.join(lines))
