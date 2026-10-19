"""Replay cached held-out predictions through fresh causal student inference."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from inference import StudentInference
from model import Student
from pilot import ROOT
from predict import detector_scores
from infer_video import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(8)
    m=json.loads((a.root/'manifest.json').read_text());record=next(r for r in m['sequences'] if r['split']=='heldout')
    seq=torch.load(record['path'],map_location='cuda',weights_only=True);ck=torch.load(a.root/'distilled-best.pt',map_location='cuda',weights_only=False)
    reference=Student().cuda().eval();reference.load_state_dict(ck['model']);runner=StudentInference(a.root,a.root/'yolo-settings');state=None;errors=[];times=[]
    with torch.inference_mode():
        for row in seq:
            logits,state=reference(row['roi'].float(),row['scene'].float(),row['boxes'],row['agents'],state)
            expected=detector_scores(logits.sigmoid(),row['q']).cpu().numpy()
            image=np.array(Image.open(ROOT/'rgb-images'/record['video']/f"{row['fid']:05d}.jpg").convert('RGB'))
            torch.cuda.synchronize();start=time.perf_counter();actual=runner.predict_frame(image);torch.cuda.synchronize();times.append(time.perf_counter()-start)
            assert np.array_equal(actual['boxes'],row['boxes'].cpu().numpy())
            assert np.array_equal(actual['confidence'],row['q'].cpu().numpy())
            assert np.allclose(expected,actual['scores'],atol=1e-4,rtol=1e-3)
            errors.append(float(np.abs(expected-actual['scores']).max(initial=0)))
    result={'passed':True,'video':record['video'],'frames':len(seq),'max_abs_score_error':max(errors),'frame_seconds':times,'scope':'fresh detector/backbone/causal state vs stored heldout features; one8-frame segment'}
    write_json(a.root/'runtime-verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
