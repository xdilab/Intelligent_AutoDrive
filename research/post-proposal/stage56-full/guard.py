import json
from pathlib import Path
root=Path('/work/bbyrd1/stage56-encoder-pilot-20260914/results')
for stage in ['stage5','stage6']:
 d=json.loads((root/f'smoke-{stage}-contrastive.json').read_text());assert d['passed'] and d['inference_parity_max_abs_error']<=1e-6 and d['frozen_sha256_before']==d['frozen_sha256_after']
 assert min(d[k] for k in ['classification_visual_gradient_norm','contrastive_visual_gradient_norm','composition_mlp_gradient_norm','vision_parameter_update_l2'])>0
 if stage=='stage6':assert d['phrase_branch_gradient_norm']>0

recovery=json.loads(Path("/work/bbyrd1/stage56-full-20260914/results/frame-recovery.json").read_text());assert recovery["passed"] and recovery["frame_root"]=="/work/bbyrd1/stage56-full-20260914/frames"
