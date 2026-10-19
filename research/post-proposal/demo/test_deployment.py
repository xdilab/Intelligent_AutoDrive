"""GPU residency, empty-scene and original preprocessing contract checks."""
import unittest
import numpy as np
import torch

from deployment import extract_gpu, score_features
from runtime import extract


class DeterministicEncoder:
    def __call__(self, clips, key_t):
        values=clips[:,key_t].mean((1,2,3))
        return values[:,None,None].expand(-1,256,1408).contiguous()


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class DeploymentTest(unittest.TestCase):
    def test_device_features_preserve_reference_rounding(self):
        images=np.random.default_rng(51).integers(0,256,(8,64,96,3),dtype=np.uint8)
        encoder=DeterministicEncoder()
        for boxes,key in [(np.array([[.1,.2,.5,.9],[0,0,.01,.01],[.8,.8,1,1]],np.float32),3),
                          (np.empty((0,4),np.float32),0)]:
            reference=extract(encoder,images,boxes,key,2)
            resident=extract_gpu(encoder,images,boxes,key,2)
            for name in reference:
                self.assertEqual(resident[name].device.type,'cuda')
                torch.testing.assert_close(resident[name].cpu(),reference[name],rtol=0,atol=0)

    def test_no_detections_returns_empty_scores_without_calling_heads(self):
        class UnusedHead:
            def __call__(self,*args):
                raise AssertionError('No candidate should enter a head')
        features={'boxes':torch.empty((0,4),device='cuda')}
        for scores in score_features([UnusedHead(),UnusedHead()],.5,features,[]).values():
            self.assertEqual(tuple(scores.shape),(0,184))
            self.assertEqual(scores.device.type,'cuda')

    def test_invalid_detector_boxes_fail_before_encoding(self):
        images=np.zeros((8,64,96,3),np.uint8)
        for box in [[.8,0,.1,1],[0,0,1.1,1],[0,0,float('nan'),1]]:
            with self.assertRaisesRegex(ValueError,'Invalid normalized'):
                extract_gpu(None,images,[box])


if __name__=='__main__':unittest.main()
