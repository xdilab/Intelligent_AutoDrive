import unittest
import torch
from model import objective

class AllLabels(unittest.TestCase):
    def test_each_group_directly_supervised(self):
        # Agentness, agent, action, location, duplex, and both ends of triplet range.
        indices=[0,1,11,33,49,98,183]
        y=torch.zeros(len(indices),184)
        for row,col in enumerate(indices):y[row,col]=1
        similarity=torch.zeros_like(y,requires_grad=True)
        _,_,loss=objective({'logits':torch.zeros_like(y),'contrastive_logits':similarity},y,torch.ones(184)*.5)
        loss.backward()
        for row,col in enumerate(indices):
            self.assertLess(similarity.grad[row,col].item(),0)
            self.assertGreater(similarity.grad[row,(col+1)%184].item(),0)
            torch.testing.assert_close(similarity.grad[row,col],torch.tensor((1/184-1)/len(indices)))
    def test_multilabel_positive_normalization_and_background(self):
        y=torch.zeros(3,184);y[0,[0,2,14,40,60,110]]=1;y[1,[11,12]]=1
        s=torch.randn(3,184,requires_grad=True)
        _,_,loss=objective({'logits':torch.zeros_like(y),'contrastive_logits':s},y,torch.ones(184)*.5)
        expected=-(s[0].log_softmax(0)[[0,2,14,40,60,110]].mean()+s[1].log_softmax(0)[[11,12]].mean())/2
        torch.testing.assert_close(loss,expected);loss.backward();self.assertEqual(s.grad[2].abs().sum().item(),0)
    def test_empty_batch_of_positives_is_safe(self):
        y=torch.zeros(3,184);s=torch.randn_like(y,requires_grad=True)
        _,_,loss=objective({'logits':torch.zeros_like(y),'contrastive_logits':s},y,torch.ones(184)*.5)
        self.assertEqual(loss.item(),0);loss.backward();self.assertEqual(s.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()
