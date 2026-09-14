import ast,os,json,hashlib
from pathlib import Path
import torch
root=Path(os.environ['COMPARISON_ROOT']);code=root/'code'
s=(code/'cache-paired.py').read_text();tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_clip_s')
REPO='OpenGVLab/InternVideo2_CLIP_S';REV='1f9fca1389fd883defc652634d95a21121c85a8c';ADAPTED=Path(os.environ['ADAPTED_CHECKPOINT']);exec(compile(ast.Module(body=[fn],type_ignores=[]),'loader','exec'))
labels=json.loads((code/'shared-frames.json').read_text())['labels'];phrases=json.loads((code/'phrases.json').read_text());order=['agentness'];texts=[phrases['agentness']]
for h in ['agent','action','loc','duplex','triplet']:
 for label in labels[h]:order.append(h+':'+label);texts.append(phrases[h][label])
assert len(texts)==184
for condition in ['original','adapted']:
 model=load_clip_s('cuda',condition=='adapted')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):emb=model.encode_text(model.tokenizer(texts).cuda()).float()
 emb=torch.nn.functional.normalize(emb,dim=-1).cpu();assert emb.shape==(184,512) and torch.isfinite(emb).all()
 torch.save({'embeds':emb,'order':order,'revision':REV,'adapted_checkpoint_sha256':hashlib.sha256(ADAPTED.read_bytes()).hexdigest() if condition=='adapted' else None},root/'data'/condition/'phrase_embeds.pt')
 del model;torch.cuda.empty_cache()
old=torch.load('/work/bbyrd1/proposal-study-20260910/data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds']
new=torch.load(root/'data/original/phrase_embeds.pt',weights_only=False)['embeds'];assert torch.allclose(old,new,atol=.002,rtol=.002),'Original phrase embeddings changed unexpectedly'
print('Verified both 184x512 phrase matrices and original-embedding parity.',flush=True)
