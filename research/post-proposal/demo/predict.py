"""Run the actual scaled CSEB experts on a verified portable feature cache."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from export_sample import sha

OFFSETS = {'action': (11, 33), 'loc': (33, 49), 'duplex': (49, 98), 'triplet': (98, 184)}


def blend_scores(a, b, weight):
    if not 0 <= weight <= 1 or a.shape != b.shape or a.shape[-1] != 184:
        raise ValueError('Expected matching 184-column scores and weight in [0,1]')
    return torch.cat((a[:, :49], (1-weight)*a[:, 49:] + weight*b[:, 49:]), dim=1)


def detector_scores(probabilities, confidence):
    result = probabilities * confidence[:, None]
    result[:, 0] = confidence
    return result


def load_experts(checkpoint, model_path, manifest, device):
    # Only load checkpoints produced by this repository, never untrusted uploads.
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck.get('kind') != 'blend' or len(ck['models']) != 2:
        raise ValueError('Expected a two-expert AP-selected blend checkpoint')
    signature = ck['signature']
    for key in ['protocol_sha256', 'cache_marker_sha256']:
        manifest_key = 'training_cache_marker_sha256' if key=='cache_marker_sha256' and manifest.get('source_type')=='raw-video' else key
        if signature[key] != manifest[manifest_key]:
            raise ValueError(f'Checkpoint/cache mismatch: {key}')
    if sha(model_path) != signature['code_sha256']['model.py']:
        raise ValueError('Model implementation differs from trained checkpoint')
    spec = importlib.util.spec_from_file_location('cseb_exact_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bank = ck['models'][0]['phrase_bank']
    if not torch.equal(bank, ck['models'][1]['phrase_bank']):
        raise ValueError('Experts must use the same phrase bank')
    models = []
    for state in ck['models']:
        model = module.ContextualRoIHead(bank, fusion='attention')
        model.load_state_dict(state, strict=True)
        models.append(model.to(device).eval())
    return models, float(ck['weight']), signature


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'contextual-roi-1b/model.py')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--display-floor', type=float, default=.1)
    a = ap.parse_args()
    if a.batch_size < 1 or not 0 <= a.display_floor <= 1:
        ap.error('Invalid batch size or display floor')
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.sample / 'manifest.json').read_text())
    for name in ['q', 'agent']:
        if sha(a.sample / f'{name}.npy') != manifest[f'{name}_sha256']:
            raise ValueError(f'{name} checksum mismatch')
    q = np.load(a.sample / 'q.npy', allow_pickle=False)
    agents = np.load(a.sample / 'agent.npy', allow_pickle=False)
    models, weight, signature = load_experts(a.checkpoint, a.model, manifest, a.device)
    frames = []
    raw_a, raw_b, raw_blend = [], [], []
    start = time.monotonic()
    for index, record in enumerate(manifest['frames']):
        path = a.sample / 'cache' / (record['key'] + '.pt')
        if sha(path) != record['cache_sha256']:
            raise ValueError('Cache checksum mismatch')
        cache = torch.load(path, map_location='cpu', weights_only=True)
        n = record['row_end'] - record['row_start']
        expected = {'crop': (n,1408), 'context_roi': (n,1408), 'scene_tokens': (16,1408), 'boxes': (n,4)}
        for key, shape in expected.items():
            if tuple(cache[key].shape) != shape or not torch.isfinite(cache[key]).all():
                raise ValueError(f'Invalid {key} cache')
        scores = [[], []]
        with torch.inference_mode():
            for j in range(0, n, a.batch_size):
                end = min(j+a.batch_size, n)
                xs = [cache['crop'][j:end], cache['context_roi'][j:end],
                      cache['scene_tokens'][None].expand(end-j,-1,-1), cache['boxes'][j:end]]
                xs = [x.to(a.device) for x in xs]
                with torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device=='cuda'):
                    for k, model in enumerate(models):
                        scores[k].append(model(*xs)['logits'].float().sigmoid().cpu())
        pa, pb = [torch.cat(s) if s else torch.empty(0,184) for s in scores]
        mixed = blend_scores(pa, pb, weight)
        if not torch.isfinite(mixed).all():
            raise ValueError('Nonfinite predictions')
        confidence = torch.from_numpy(np.array(q[record['row_start']:record['row_end']], dtype=np.float32))
        p = detector_scores(mixed, confidence)
        sa, sb = detector_scores(pa, confidence), detector_scores(pb, confidence)
        raw_a.append(sa.numpy()); raw_b.append(sb.numpy()); raw_blend.append(p.numpy())
        detections = []
        for i in torch.argsort(confidence, descending=True).tolist():
            if confidence[i] < a.display_floor:
                continue
            agent = int(agents[record['row_start']+i])
            if not 0 <= agent < len(manifest['labels']['agent']):
                raise ValueError('Invalid YOLO agent class')
            heads = {}
            for head, (lo, hi) in OFFSETS.items():
                ix = sorted(set(k for tensor in (p,sa,sb)
                                for k in torch.argsort(tensor[i,lo:hi], descending=True)[:3].tolist()))
                heads[head] = [{'label':manifest['labels'][head][k], 'score':float(p[i,lo+k]),
                                'classification':float(sa[i,lo+k]), 'contrastive':float(sb[i,lo+k])} for k in ix]
            detections.append({'candidate':i, 'box':cache['boxes'][i].tolist(),
                               'confidence':float(confidence[i]), 'agent':manifest['labels']['agent'][agent],
                               'heads':heads})
        frames.append({'key':record['key'], 'fid':record['fid'], 'video':record['video'],
                       'input_fids':record['input_fids'], 'candidate_count':n, 'detections':detections})
        if index == 0 or (index+1)%10 == 0:
            print(json.dumps({'frames':index+1,'total':len(manifest['frames']), 'seconds':round(time.monotonic()-start,2)}),flush=True)
    elapsed = time.monotonic()-start
    np.savez_compressed(a.output / 'scores.npz', classification=np.concatenate(raw_a),
                        contrastive=np.concatenate(raw_b), blend=np.concatenate(raw_blend))
    report = {'format':1, 'model':'CSEB · scaled scene-attention expert blend',
              'run':a.checkpoint.parent.name, 'weight':weight, 'frames':frames,
              'display_floor':a.display_floor, 'inference_seconds':elapsed,
              'device':a.device, 'precision':'bfloat16 autocast' if a.device=='cuda' else 'float32',
              'checkpoint_sha256':sha(a.checkpoint), 'model_sha256':sha(a.model),
              'manifest_sha256':sha(a.sample/'manifest.json'), 'signature':signature,
              'mode':('End-to-end raw video: fresh YOLO detections and fresh visual-encoder features' if manifest.get('source_type')=='raw-video'
                      else 'Head inference on cached frozen-encoder features and YOLO proposals'),
              'source_type':manifest.get('source_type','study-cache'),
              'detector_settings':manifest.get('detector'),
              'selection':manifest['selection'], 'source_split':manifest['split'],
              'note':'Qualitative single-seed demonstration; no AP computed. Agent labels come from YOLO. Scores include detector confidence. CPU float32 may differ slightly from published GPU bfloat16 inference.'}
    (a.output / 'predictions.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'complete':True,'seconds':elapsed,'frames':len(frames)}))


if __name__ == '__main__':
    main()
