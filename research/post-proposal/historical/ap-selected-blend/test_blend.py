"""Selection rules and actual evaluator endpoint parity on retained synthetic fixture."""
import ast,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from selection import choose,validate_selection
A=Path(__file__).resolve().parent
assert choose([{'weight':.75,'dev_triplet_crop_AP':2},{'weight':.25,'dev_triplet_crop_AP':2}])['weight']==.25
assert choose([{'weight':.25,'dev_triplet_crop_AP':1},{'weight':.75,'dev_triplet_crop_AP':2}])['weight']==.75
# Detection matching function is identical to tested original evaluator.
def matcher(p):return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='one_class'),include_attributes=False)
assert matcher(A/'evaluate.py')==matcher(A.parent/'class-gate-study/evaluate.py')
fixture=Path(json.loads((A.parent/'class-gate-study/test-result.json').read_text())['synthetic_root']);assert fixture.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for endpoint in [0.,1.]:
 root=Path(tempfile.mkdtemp(prefix='ap-blend-test-'));(root/'code').mkdir();sel=json.loads((A/'selection.json').read_text());sel['grid']=[0.,1.]
 for seed in sel['seeds'].values():
  for v in seed.values():v['candidates']=[{'weight':g,'dev_triplet_crop_AP':float(g==endpoint)} for g in sel['grid']];v['selected_weight']=endpoint
 validate_selection(sel);sp=root/'code/selection.json';sp.write_text(json.dumps(sel));cfg=json.loads((A/'protocol.json').read_text());cfg.update(source_study=str(fixture/'source'),expert_study=str(fixture),root=str(root),smoke=True,selection_sha256=sha(sp),training_report_sha256={'0':sha(fixture/'results/train-seed0.json')});(root/'code/protocol.json').write_text(json.dumps(cfg))
 r=subprocess.run([sys.executable,str(A/'evaluate.py'),'--root',str(root),'--seed','0','--workers','2'],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
 result=json.loads((root/'results/metrics/seed0-ap-blend.json').read_text());reference=json.loads((fixture/f'results/metrics/seed0-{"stage5" if endpoint==0 else "head-phrase"}.json').read_text())
 for group in ['duplex','triplet']:assert result['ap_values'][group]==reference['ap_values'][group]
 assert result['candidate_sha256']==reference['candidate_sha256'];assert result['selected_weight']==endpoint
(A/'test-result.json').write_text(json.dumps({'passed':True,'checks':['AP selection and deterministic lower-weight tie rule','matching routine AST parity with original','actual evaluator mixture endpoints recover Stage5/phrase composition AP','fixed candidate hash and selected-weight provenance'],'synthetic_only':True},indent=2))
print('PASS: selection and actual evaluator endpoint parity')
