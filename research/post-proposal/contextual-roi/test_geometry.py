import unittest
import torch
from cache_context import contextual_features

class Geometry(unittest.TestCase):
    def test_full_frame_and_actor_location(self):
        # x-gradient must yield larger actor feature for the right-hand actor.
        x=torch.arange(16).float()[None,None,None,:].expand(1,4,16,16)
        boxes=torch.tensor([[0.,0.,.4,1.],[.6,0.,1.,1.]])
        roi,scene=contextual_features(x,boxes)
        self.assertEqual(roi.shape,(2,4));self.assertEqual(scene.shape,(16,4))
        self.assertTrue((roi[1]>roi[0]+5).all())
        full,_=contextual_features(torch.ones_like(x),torch.tensor([[0.,0.,1.,1.]]))
        torch.testing.assert_close(full,torch.ones(1,4))

    def test_zero_area_boundary_preserves_candidate(self):
        from model import ContextualRoIHead
        boxes=torch.tensor([[.978125,1.,1.,1.],[0.,0.,1.,1.]])
        roi,scene=contextual_features(torch.ones(1,4,16,16),boxes)
        self.assertEqual(roi.shape,(2,4))
        torch.testing.assert_close(roi[0],torch.zeros(4))
        torch.testing.assert_close(roi[1],torch.ones(4))
        geometry=ContextualRoIHead.position(boxes)
        self.assertEqual(geometry[0,-1].item(),0.)
        self.assertTrue(torch.isfinite(geometry).all())

if __name__=='__main__':unittest.main()
