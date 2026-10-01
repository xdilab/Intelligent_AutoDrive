"""Frozen-cache residual actor/context fusion. No encoder or labels in forward()."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, dim, bottleneck=128):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, bottleneck),
                                 nn.GELU(), nn.Linear(bottleneck, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class ContextualRoIHead(nn.Module):
    def __init__(self, phrase_bank, visual_dim=1408, dim=512, fusion='mlp', heads=8):
        super().__init__()
        if phrase_bank.shape != (184, 768):
            raise ValueError('Expected ordered184x768 phrase bank from verified1B text tower.')
        if fusion not in ('mlp', 'attention'):
            raise ValueError(fusion)
        self.fusion = fusion
        self.register_buffer('phrase_bank', F.normalize(phrase_bank.detach().float(), dim=-1))
        self.text_input_projection = nn.Linear(768, dim)
        self.crop_adapter = ResidualAdapter(visual_dim)
        self.text_adapter = ResidualAdapter(dim)
        self.crop_projection = nn.Linear(visual_dim, dim)
        self.context_projection = nn.Linear(visual_dim, dim)
        self.scene_projection = nn.Linear(visual_dim, dim)
        self.geometry = nn.Sequential(nn.Linear(8, 128), nn.GELU(), nn.Linear(128, dim))
        self.visual_fusion = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.context_attention = nn.MultiheadAttention(dim, heads, batch_first=True) if fusion == 'attention' else None
        self.visual_norm = nn.LayerNorm(dim)
        self.language_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.language_scale = nn.Parameter(torch.tensor(-2.0))
        self.joint_norm = nn.LayerNorm(dim)
        # Joint RoI + residual visual stream + language retrieval scores.
        self.classifier = nn.Sequential(nn.Linear(dim * 2 + 184, dim), nn.GELU(), nn.Linear(dim, 184))

    @staticmethod
    def position(boxes):
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError('boxes must be N x 4 normalized xyxy')
        if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
            raise ValueError('Nonfinite/out-of-range boxes')
        wh = boxes[:, 2:] - boxes[:, :2]
        if (wh < 0).any():
            raise ValueError('Inverted boxes')
        return torch.cat([boxes, (boxes[:, :2] + boxes[:, 2:]) / 2, wh], -1)

    def forward(self, crop, context_roi, scene, boxes):
        """crop/context_roi: N,D; scene: N,K,D; boxes: N,4. All cache tensors detached."""
        crop, context_roi, scene, boxes = [v.detach().float() for v in (crop, context_roi, scene, boxes)]
        if scene.ndim != 3 or scene.shape[0] != len(crop):
            raise ValueError('scene must contain spatial context tokens per actor')
        c = self.crop_projection(self.crop_adapter(crop))
        r = self.context_projection(context_roi)
        tokens = self.scene_projection(scene)
        g = self.geometry(self.position(boxes))
        if self.fusion == 'mlp':
            evidence = self.visual_fusion(torch.cat([c, r, tokens.mean(1), g], -1))
        else:
            attended, _ = self.context_attention((c + g).unsqueeze(1), tokens, tokens, need_weights=False)
            evidence = self.visual_fusion(torch.cat([c, r, attended[:, 0], g], -1))
        visual_roi = self.visual_norm(c + evidence)
        text = self.text_adapter(self.text_input_projection(self.phrase_bank))
        bank = text.unsqueeze(0).expand(len(crop), -1, -1)
        language, _ = self.language_attention(visual_roi[:, None], bank, bank, need_weights=False)
        joint_roi = self.joint_norm(visual_roi + self.language_scale.sigmoid() * language[:, 0])
        phrase_scores = F.normalize(visual_roi, dim=-1) @ F.normalize(text, dim=-1).T
        logits = self.classifier(torch.cat([joint_roi, c, phrase_scores], -1))
        # Align both learned branches before language fusion; no text input enters visual_roi.
        contrastive_logits = F.normalize(visual_roi, dim=-1) @ F.normalize(text, dim=-1).T / .07
        return {'logits': logits, 'roi_features': joint_roi, 'visual_roi_features': visual_roi,
                'contrastive_logits': contrastive_logits}


def objective(outputs, targets, alpha, contrastive_weight=.001):
    """All 184 labels focal; auxiliary multi-positive contrastive over all184 label phrases."""
    z = outputs['logits'].float()
    p = z.sigmoid()
    pt = targets * p + (1 - targets) * (1 - p)
    classification = ((targets * alpha + (1 - targets) * (1 - alpha)) * (1 - pt).square()
                      * F.binary_cross_entropy_with_logits(z, targets, reduction='none')).mean()
    y = targets
    sim = outputs['contrastive_logits']
    if y.shape != sim.shape or y.shape[-1] != 184:
        raise ValueError('All-label contrastive objective requires aligned184-column targets and logits')
    valid = y.sum(-1) > 0
    contrastive = -(y[valid] * sim[valid].log_softmax(-1)).sum(-1).div(y[valid].sum(-1)).mean() if valid.any() else sim.sum() * 0
    return classification + contrastive_weight * contrastive, classification, contrastive
