"""Stage 7: contextual residual features into frozen seed-matched Stage 5 heads."""
from pathlib import Path
import json
import torch
from torch import nn
from contextual import ContextualRoIHead, objective

class Stage7(nn.Module):
    def __init__(self, bank, flat, comp, visual_dim=1024, dim=512, baseline=False):
        super().__init__()
        self.context=ContextualRoIHead(bank,visual_dim=visual_dim,dim=dim,fusion='mlp')
        # Remove the standalone classifier. Its computed scores are never input to Stage5.
        self.context.classifier=nn.Identity()
        self.bridge=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,visual_dim))
        nn.init.zeros_(self.bridge[-1].weight);nn.init.zeros_(self.bridge[-1].bias)
        self.flat=flat.requires_grad_(False);self.comp=comp.requires_grad_(False)
        self.baseline=baseline

    def assemble(self, x):
        # Preserve original Stage5 FP32 readout, including gradients to its input.
        with torch.autocast(x.device.type,enabled=False):
            raw=self.flat(x.float())
            compositions=self.comp(torch.cat([raw[:,:49].sigmoid(),x.float()],-1))
            return torch.cat([raw[:,:49],compositions],-1)

    def forward(self,crop,context_roi,scene,boxes):
        x=crop.detach().float()
        if self.baseline:
            return {'logits':self.assemble(x),'refined_features':x}
        features=self.context(crop,context_roi,scene,boxes)
        delta=self.bridge(features['roi_features']).float()
        refined=x+delta
        return {'logits':self.assemble(refined), 'contrastive_logits':features['contrastive_logits'],
                'refined_features':refined,'delta':delta,'roi_features':features['roi_features']}

def make_model(root,seed,baseline=False):
    root=Path(root);run=root/'inputs'/f'seed-{seed}'
    f=torch.load(run/'head-flat.pt',map_location='cpu',weights_only=False)
    c=torch.load(run/'stage5.pt',map_location='cpu',weights_only=False)
    assert f['kind']=='flat' and c['in_dim']==1073 and f['seed']==c['seed']==seed
    assert f['expert_videos']==c['expert_videos'] and len(f['expert_videos'])==420
    flat=nn.Linear(1024,184);flat.load_state_dict(f['state'])
    comp=nn.Sequential(nn.Linear(1073,512),nn.ReLU(),nn.Linear(512,135));comp.load_state_dict(c['state'])
    bank=torch.load(root/'data/phrase_embeds.pt',map_location='cpu',weights_only=False)['embeds']
    m=Stage7(bank,flat,comp,baseline=baseline)
    cfg=json.loads((root/'protocol.json').read_text());p=Path(cfg['context_source'])/'runs'/f'mlp-contrastive-all184-seed{seed}'/'best.pt'
    ck=torch.load(p,map_location='cpu',weights_only=False)
    assert ck['seed']==seed and ck['fusion']=='mlp' and ck['signature']['run']['contrastive_target']=='adapted_text'
    weights={k:v for k,v in ck['model'].items() if not k.startswith('classifier.')}
    m.context.load_state_dict(weights,strict=True)
    return m
