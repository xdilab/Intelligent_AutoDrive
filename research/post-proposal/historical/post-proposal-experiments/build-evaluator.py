"""Make an isolated, manifest-enforcing copy; never edit the source repository."""
from pathlib import Path
import hashlib,json
root=Path(__file__).parent
src=Path('/data/repos/ROAD_Reason/experiments/exp12_phrase_head/eval_comb.py')
s=src.read_text()
s=s.replace('E12 = Path(__file__).resolve().parent','import os\nE12 = Path(os.environ.get("ROAD_E12", "/data/repos/ROAD_Reason/experiments/exp12_phrase_head"))')
s=s.replace('ap.add_argument("--ckpt", required=True)','ap.add_argument("--manifest", required=True)\nap.add_argument("--ckpt", required=True)')
s=s.replace('n_frames = 0; t0 = time.time()','manifest = json.loads(Path(args.manifest).read_text())\nframe_keys = manifest["frames"]\nassert len(frame_keys) == len(set(frame_keys)), "Duplicate manifest frames"\nempty_feature_frames = []\nn_frames = 0; t0 = time.time()')
s=s.replace('for stem in sorted(yolo):','for stem in frame_keys:')
s=s.replace('''        if key not in i3d or fkey not in feats:
            continue
        yrec = yolo[stem]
        yb = yrec["boxes_xyxyn"].astype(np.float32)
        f = feats[fkey]
        if f.shape[0] != yb.shape[0]:
            continue
        f32 = np.nan_to_num(f.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)''','''        assert stem in yolo and key in i3d, f"Missing detector/GT record: {stem}"
        yrec = yolo[stem]
        yb = yrec["boxes_xyxyn"].astype(np.float32)
        if fkey in feats:
            f = feats[fkey]
        elif len(yb) == 0:
            f = np.empty((0, FD), dtype=np.float16)
            empty_feature_frames.append(stem)
        else:
            raise ValueError(f"Missing features for nonempty frame: {stem}")
        assert f.shape == (len(yb), FD), f"Feature row/dimension mismatch: {stem}"
        assert np.isfinite(f).all(), f"Nonfinite features: {stem}"
        f32 = f.astype(np.float32)''')
s=s.replace('"n_frames": n_frames,','"n_frames": n_frames, "frame_sha256": manifest["frame_sha256"],\n     "candidate_sha256": manifest["candidate_sha256"], "empty_feature_frames": empty_feature_frames,')
assert 'if key not in i3d or fkey not in feats:' not in s
(root/'eval-crops-shared.py').write_text(s)
(root/'evaluator-provenance.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'changes':['explicit fixed frame manifest','empty candidates get an empty feature matrix and retain GT/frame','missing nonempty features, mismatched rows, and nonfinite features fail instead of silently dropping frames','explicit source paths; original repository unchanged'],'evaluator':'Original baseline modules.evaluation.evaluate at IoU 0.5; unchanged'},indent=2)+'\n')
