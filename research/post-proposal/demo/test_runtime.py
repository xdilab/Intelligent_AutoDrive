"""Verify video sampling and preprocessing against the original training code."""
import ast
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align

from infer_video import VideoReader
from runtime import normalize_clip,crop_batch


class VideoSamplingTest(unittest.TestCase):
    def test_rate_and_boundary_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(96,64))
            for i in range(20):writer.write(np.full((64,96,3),i*10,np.uint8))
            writer.release();reader=VideoReader(path)
            self.assertEqual(reader.sample_count,10)
            images,fids,key=reader.clip(1)
            self.assertEqual(fids,[1,1,1,1,2,3,4,5])
            self.assertEqual(key,0)
            self.assertEqual(images.shape,(8,64,96,3))
            np.testing.assert_array_equal(images[0],images[3])
            self.assertLess(abs(float(images[4].mean())-20),3)
            _,last,_=reader.clip(10)
            self.assertEqual(last,[7,8,9,10,10,10,10,10])
            reader.cap.release()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for exact preprocessing comparison')
    def test_original_preprocessing_parity(self):
        path=Path(__file__).resolve().parents[1]/'stage56-full/eval_input.py'
        tree=ast.parse(path.read_text())
        defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_clip','crop_batch']]
        namespace={'torch':torch,'np':np,'roi_align':roi_align}
        exec(compile(ast.Module(body=defs,type_ignores=[]),str(path),'exec'),namespace)
        images=np.random.default_rng(42).integers(0,256,(8,64,96,3),dtype=np.uint8)
        boxes=np.array([[.1,.2,.5,.9],[0,0,.01,.01],[.8,.8,1,1]],np.float32)
        actual=normalize_clip(images);reference=namespace['normalize_clip'](images)
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        torch.testing.assert_close(crop_batch(actual,boxes),namespace['crop_batch'](reference,boxes),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
