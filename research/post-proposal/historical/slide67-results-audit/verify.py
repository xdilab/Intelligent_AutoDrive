import json,hashlib,datetime
from pathlib import Path
A=Path(__file__).resolve().parent;W=A.parent;R=Path('/data/repos/ROAD_Reason/experiments');live=json.loads((A/'live-slide67.json').read_text())
def celltext(cell):return ''.join(e.get('textRun',{}).get('content','') for e in cell.get('text',{}).get('textElements',[])).strip()
table=next(e['table'] for e in live['pageElements'] if 'table' in e);rows=[[celltext(c) for c in r['tableCells']] for r in table['tableRows']];print(rows[0]);counts=json.loads((W/'proposal-tail/train-counts.json').read_text());deep=[r['label'] for r in counts['rows'] if r['z']<-.5];assert len(deep)==28
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'slide_url':'https://docs.google.com/presentation/d/1_ywXj_hqlgdC0S68aH3kGAteT5UQ4oH3Q5-Pg6s_TeM/edit#slide=id.g3fa051c7877_0_34','deep_labels':deep,'slide_rows':rows,'verified':[]}
files=list((R/'exp11_yolo').glob('results*.json'))+list((R/'exp12_phrase_head/crop_full').glob('results_crop_full*.json'))
for row in rows[1:]:
 vals=[float(x.replace('*','')) for x in row[2:]];matches=[]
 for path in files:
  d=json.loads(path.read_text());sm=d.get('summary',{});groups=['agentness','agent','action','loc','duplex','triplet']
  if not all(k in sm for k in groups):continue
  if not all(abs(sm[k]-v)<.0051 for k,v in zip(groups,vals)):continue
  ap={s.split(' : ')[0]:float(s.split(' : ')[-1]) for s in d['per_class']['triplet']};value=sum(ap[l] for l in deep)/28;assert abs(value-vals[-1])<.0051
  matches.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n_frames':d.get('n_frames'),'deep28_recomputed':value,'summary':sm})
 assert matches,(row,'No matching raw result');report['verified'].append({'stage':row[0],'matches':matches});print(row[0],[(Path(m['source']).name,m['deep28_recomputed'],m['n_frames']) for m in matches])
# Independently recompute the latest quoted numbers, not just their aggregate JSON.
recent={}
for variant in ['head-phrase','stage5','stage6']:
 vals=[]
 for seed in range(3):
  p=W/f'class-gate-study/collected/metrics/seed{seed}-{variant}.json';d=json.loads(p.read_text());ap={s.split(' : ')[0]:float(s.split(' : ')[-1]) for s in d['per_class']['triplet']};v=sum(ap[l] for l in deep)/28;assert abs(v-d['tail']['deep28']['mAP'])<1e-5;vals.append(v)
 recent[variant]={'per_seed':vals,'mean':sum(vals)/3}
report['new_70_percent_study']=recent;(A/'verification.json').write_text(json.dumps(report,indent=2));print(recent)
