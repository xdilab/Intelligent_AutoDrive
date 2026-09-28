import unittest
import torch
from model import ContextualRoIHead, objective

class Contracts(unittest.TestCase):
    def test_two_sided_gradients_and_separate_visual_path(self):
        torch.set_num_threads(2)
        for fusion in ('mlp', 'attention'):
            torch.manual_seed(3)
            m = ContextualRoIHead(torch.randn(184, 32), visual_dim=64, dim=32, fusion=fusion, heads=4)
            inputs = [torch.randn(4,64,requires_grad=True), torch.randn(4,64,requires_grad=True),
                      torch.randn(4,16,64,requires_grad=True), torch.tensor([[.1,.2,.5,.7]]*4)]
            y = torch.zeros(4,184); y[:3,0]=1; y[0,98]=1; y[0,99]=1; y[1,100]=1; y[2,101]=1
            out = m(*inputs)
            self.assertEqual(out['logits'].shape, (4,184))
            loss, cls, aux = objective(out,y,torch.full((184,),.5))
            aux.backward(retain_graph=True)
            self.assertTrue(m.context_projection.weight.grad.abs().sum()>0)
            self.assertTrue(m.text_adapter.net[-1].weight.grad.abs().sum()>0)
            self.assertTrue(m.visual_fusion[-1].weight.grad.abs().sum()>0)
            self.assertTrue(all(p.grad is None for p in m.language_attention.parameters()))
            self.assertTrue(all(p.grad is None for p in m.classifier.parameters()))
            self.assertFalse(m.phrase_bank.requires_grad)
            self.assertTrue(all(v.grad is None for v in inputs[:3]))
            m.zero_grad(); loss.backward()
            self.assertTrue(m.text_adapter.net[-1].weight.grad.abs().sum()>0)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(objective(m(*inputs),torch.zeros_like(y),torch.full((184,),.5))[2].item(),0)
            # Adapted text changes alignment scores, but cannot change pre-fusion visual features.
            before=m(*inputs)
            with torch.no_grad(): m.text_adapter.net[-1].bias.add_(torch.randn(32))
            after=m(*inputs)
            torch.testing.assert_close(before['visual_roi_features'],after['visual_roi_features'])
            self.assertFalse(torch.allclose(before['contrastive_logits'],after['contrastive_logits']))
            self.assertFalse(torch.allclose(before['logits'],after['logits']))

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError): ContextualRoIHead.position(torch.tensor([[.5,.1,.4,.8]]))

if __name__=='__main__':unittest.main()
