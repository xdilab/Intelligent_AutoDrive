"""Protocol guards for the demo port, independent of private datasets."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from export_sample import slice_npy
from predict import blend_scores, detector_scores


class DemoProtocolTest(unittest.TestCase):
    def test_primitive_scores_are_never_blended(self):
        a=torch.full((3,184),.2);b=torch.full((3,184),.8)
        mixed=blend_scores(a,b,.75)
        torch.testing.assert_close(mixed[:,:49],a[:,:49])
        torch.testing.assert_close(mixed[:,49:],torch.full((3,135),.65))

    def test_zero_weight_recovers_classification_expert(self):
        a=torch.rand(2,184);b=torch.rand(2,184)
        torch.testing.assert_close(blend_scores(a,b,0),a)

    def test_detector_confidence_is_applied_once(self):
        p=torch.full((2,184),.8);q=torch.tensor([.5,.2])
        scores=detector_scores(p,q)
        torch.testing.assert_close(scores[:,0],q)
        torch.testing.assert_close(scores[:,1],torch.tensor([.4,.16]))
        torch.testing.assert_close(p,torch.full((2,184),.8))

    def test_invalid_blend_contract_rejected(self):
        with self.assertRaises(ValueError):blend_scores(torch.zeros(2,184),torch.zeros(2,184),1.1)
        with self.assertRaises(ValueError):blend_scores(torch.zeros(2,183),torch.zeros(2,183),.5)

    def test_portable_array_slice_preserves_dtype_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for dtype in ('float32','float64','int8','int64'):
                data=np.arange(60,dtype=dtype).reshape(10,6)
                np.save(root/'source.npy',data)
                slice_npy(root/'source.npy',root/'slice.npy',2,5)
                copied=np.load(root/'slice.npy',allow_pickle=False)
                self.assertEqual(copied.dtype,data.dtype)
                np.testing.assert_array_equal(copied,data[2:5])


if __name__=='__main__':unittest.main()
