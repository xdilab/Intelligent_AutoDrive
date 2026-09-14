"""One-time collection and wiki status update; no background polling or submission."""
import datetime,json,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent;D=A/'collected';D.mkdir(exist_ok=True);W=A.parent.parent
subprocess.run(['rsync','-rt','ncshare:/work/bbyrd1/ap-selected-blend-20260914/results/',str(D)+'/'],check=True)
jobs=json.loads((D/'jobs.json').read_text())['jobs'];status=subprocess.check_output(['ssh','ncshare','sacct -j '+','.join(jobs.values())+' -X --format=JobID,State,Elapsed,ExitCode -n'],text=True);now=datetime.datetime.now(datetime.timezone.utc).isoformat();(D/'status.txt').write_text(now+'\n'+status)
manifest=json.loads((A.parent/'post-proposal-experiments/shared-frames.json').read_text());cfg=json.loads((A/'protocol.json').read_text());metrics=list((D/'metrics').glob('*.json'))
for p in metrics:
 m=json.loads(p.read_text());assert m['n_frames']==36717 and m['frame_sha256']==manifest['frame_sha256'] and m['candidate_sha256']==manifest['candidate_sha256'] and m['protocol']==cfg and m['selection_sha256']==cfg['selection_sha256']
lines=['<!-- AP BLEND STATUS START -->','',f'Last collected: **{now}**. **{len(metrics)}/6 new detection evaluations available.**','','```text',status.strip(),'```']
if len(metrics)==6 and (D/'summary.json').exists():
 s=json.loads((D/'summary.json').read_text());assert s['n_new_evaluations']==6 and s['protocol']==cfg;lines+=['','Matched 70%-video experts. Mean ± sample SD across three seeds; detection AP percentages.','','| Variant | Triplet | Tail 47 | Deep 28 | Common 39 |','|---|---:|---:|---:|---:|']
 for r in s['means']:lines.append('| '+r['variant']+' | '+' | '.join(f"{r[k+'_mean']:.4f} ± {r[k+'_sd']:.4f}" for k in ['triplet','tail47','deep28','common39'])+' |')
lines+=['','[Collected evidence](../artifacts/ap-selected-blend/collected/). One-time collection; no continuous local monitoring implied.','','<!-- AP BLEND STATUS END -->'];p=W/'directions/class-conditioned-language-fusion.md';text=p.read_text();start=text.index('<!-- AP BLEND STATUS START -->');end=text.index('<!-- AP BLEND STATUS END -->',start)+len('<!-- AP BLEND STATUS END -->');p.write_text(text[:start]+'\n'.join(lines)+text[end:]);print('\n'.join(lines))
