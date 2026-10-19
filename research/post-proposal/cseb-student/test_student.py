import unittest
import numpy as np
import torch
from model import Student,objective
from pilot import mapped_targets,average_precision_score

class Tests(unittest.TestCase):
    def test_ap_ties(self):
        self.assertAlmostEqual(average_precision_score(np.array([1,0]),np.array([.5,.5])),.5)
        self.assertAlmostEqual(average_precision_score(np.array([1,0,1]),np.array([.9,.8,.7])),5/6)
    def test_causal_state_and_reset(self):
        torch.manual_seed(4);model=Student().eval();roi=torch.randn(3,1024);scene=torch.randn(1024);box=torch.tensor([[.1,.2,.5,.6]]*3);agent=torch.tensor([0,1,2])
        first,state=model(roi,scene,box,agent);saved=first.detach().clone()
        model(roi,scene+10,box,agent,state)
        repeated,_=model(roi,scene,box,agent)
        torch.testing.assert_close(saved,repeated,rtol=0,atol=0)
    def test_empty_frame_updates_memory(self):
        model=Student();scores,state=model(torch.empty(0,1024),torch.ones(1024),torch.empty(0,4),torch.empty(0,dtype=torch.long))
        self.assertEqual(tuple(scores.shape),(0,184));self.assertEqual(tuple(state.shape),(1,256))
    def test_multilabel_kd_gradient(self):
        scores=torch.zeros(2,184,requires_grad=True);target=torch.zeros_like(scores);target[:,[11,12,98]]=1
        objective(scores,target,target,1).backward()
        self.assertTrue((scores.grad[:,[11,12,98]]<0).all());self.assertTrue((scores.grad[:,0]==0).all())
    def test_label_remap_and_matching(self):
        labels={'agent':['Ped','Car'],'action':['Stop'],'loc':['Pav'],'duplex':['Ped-Stop'],'triplet':['Ped-Stop-Pav']}
        ann={f'all_{k}_labels':(['unused']+v) for k,v in labels.items()}
        item={'box':[0,0,1,1],**{k+'_ids':[1] for k in labels}}
        y=mapped_targets({'annos':{'a':item}},ann,labels,np.array([[0,0,1,1],[0,0,1,1]],np.float32),np.array([0,1]))
        self.assertEqual(torch.where(y[0]>0)[0].tolist(),[0,1,11,33,49,98]);self.assertEqual(float(y[1].sum()),0)
if __name__=='__main__':unittest.main()
