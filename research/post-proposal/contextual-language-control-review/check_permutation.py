import importlib.util,json,copy
from pathlib import Path
import torch
p=Path('/data/repos/ROAD_Reason/research/post-proposal/contextual-roi-all184/model.py');s=importlib.util.spec_from_file_location('source_model',p);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
torch.set_num_threads(4);torch.manual_seed(99);bank=torch.randn(184,512);m=mod.ContextualRoIHead(bank,fusion='attention').eval();n=copy.deepcopy(m)
g=torch.Generator().manual_seed(0)
while True:
 perm=torch.randperm(184,generator=g)
 if (perm!=torch.arange(184)).all():break
with torch.no_grad():
 n.phrase_bank.copy_(m.phrase_bank[perm]);n.classifier[0].weight[:,1024:].copy_(m.classifier[0].weight[:,1024:][:,perm])
 x=[torch.randn(4,1024),torch.randn(4,1024),torch.randn(4,16,1024),torch.tensor([[.1,.2,.7,.8]]*4)];a=m(*x);b=n(*x)
 d={k:float((a[k]-b[k]).abs().max()) for k in ['logits','roi_features','visual_roi_features']}
 for k in d:torch.testing.assert_close(a[k],b[k],atol=2e-6,rtol=1e-5)
result={'passed':True,'dimensions':{'phrase_bank':[184,512],'readout_input':1208},'derangement':perm.tolist(),'max_abs_difference':d,'claim':'At contrastive weight0, permuting the phrase bank is an invertible feature-coordinate relabeling. Shared rowwise text adapter plus attention is set-permutation invariant; permuting the first classifier layer phrase columns restores identical function within floating point tolerance. This is not a semantic-content removal test.','source':str(p)}
Path('/data/repos/wiki/artifacts/contextual-language-control-review/permutation-equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='derangement'},indent=2))
