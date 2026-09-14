from pathlib import Path
import json,hashlib
root=Path('/work/bbyrd1/stage56-encoder-pilot-20260914');cfg=json.loads((root/'code/protocol.json').read_text())
for stage in ['stage5','stage6']:
 d=json.loads((root/f'results/smoke-{stage}-contrastive.json').read_text());assert d['passed'] and d['protocol']==cfg
 assert d['manifest_sha256']==hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
 assert d['inference_parity_max_abs_error']<=1e-6 and d['frozen_sha256_before']==d['frozen_sha256_after']
 assert min(d[k] for k in ['classification_visual_gradient_norm','contrastive_visual_gradient_norm','composition_mlp_gradient_norm','vision_parameter_update_l2'])>0
