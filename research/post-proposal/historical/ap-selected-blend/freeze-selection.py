import json,hashlib,datetime
from pathlib import Path
from selection import choose,validate_selection
A=Path(__file__).resolve().parent;W=A.parent;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s={'locked_utc':now,'metric':'internal-development mean AP over all 86 triplets','tie_break':'lower language weight on exact AP tie','scope':'one global weight shared across 135 composition classes; independently selected for each seed and phrase/shuffled expert','seeds':{}}
for seed in range(3):
 p=W/f'gate-objective-audit/collected/audit-seed{seed}.json';d=json.loads(p.read_text());s.setdefault('grid',d['grid']);assert s['grid']==d['grid'];s['seeds'][str(seed)]={}
 for expert,e in d['experts'].items():
  cs=e['columns'];assert [x['class_index'] for x in cs]==list(range(135));candidates=[{'weight':g,'dev_triplet_crop_AP':sum(c['grid'][j]['dev_AP'] for c in cs[49:])/86} for j,g in enumerate(d['grid'])]
  s['seeds'][str(seed)][expert]={'selected_weight':choose(candidates)['weight'],'candidates':candidates,'source_report_sha256':sha(p)}
validate_selection(s);path=A/'selection.json';assert not path.exists(),'Selection is frozen; do not overwrite';path.write_text(json.dumps(s,indent=2)+'\n')
cfg={'version':'development-ap-blend-v1','locked_utc':now,'source_study':'/work/bbyrd1/proposal-study-20260910','expert_study':'/work/bbyrd1/class-gate-study-20260914','root':'/work/bbyrd1/ap-selected-blend-20260914','seeds':[0,1,2],'final_variants':['ap-blend','ap-shuffled-blend'],'selection_sha256':sha(path),'training_report_sha256':{str(seed):sha(W/f'class-gate-study/collected/train-seed{seed}.json') for seed in range(3)},'selection':'global scalar chosen from existing audit grid by internal-development triplet crop AP; exact ties favor lower language weight','final_evaluation':'same 36717 frames and YOLO candidates, AP@IoU0.5; q applied once after mixture','limitations':['same 70-percent-training-video experts and fixed partition as BCE gate study','previous final evaluation informed this follow-up; not a fresh blind test','selection uses cached-crop AP, not detector AP','no calibration or new gate training; selection-objective comparison only']}
(A/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps({k:{e:v['selected_weight'] for e,v in d.items()} for k,d in s['seeds'].items()},indent=2))
