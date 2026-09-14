import csv,datetime,hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent;W=A.parent
paths={'curves':W/'gate-objective-audit/collected/curves.csv','selection':W/'ap-selected-blend/selection.json','status':W/'ap-selected-blend/collected/status.txt','bce_summary':W/'class-gate-study/collected/summary.json','audit_summary':W/'gate-objective-audit/collected/summary.json'}
d={k:(p.read_text() if k=='status' else list(csv.DictReader(p.open())) if k=='curves' else json.loads(p.read_text())) for k,p in paths.items()};d['built_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['completed_new_evaluations']=len(list((W/'ap-selected-blend/collected/metrics').glob('*.json')))
t=(A/'template.html').read_text().replace('__DATA__',json.dumps(d).replace('</','<\\/'));(A/'seesaw-language.html').write_text(t)
(A/'provenance.json').write_text(json.dumps({'built_utc':d['built_utc'],'sources':{str(p.relative_to(W.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},'visuals':'Interactive educational seesaw and plots authored in HTML/SVG; not a replacement for draw.io stage architecture figures. Teal=Stage5, purple=phrase expert, gold=blend.','toy_example':'Synthetic 22-crop one-class example explicitly labelled; not experimental data.','snapshot':'Static build-time experiment status, not live monitoring.'},indent=2))
print(A/'seesaw-language.html')
