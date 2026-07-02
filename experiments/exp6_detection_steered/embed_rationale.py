"""Exp6 stage 2/4 — cache frozen SigLIP text embeddings of Qwen rationales.

Walks the exp5 Qwen JSON cache for every frame in a split's detection dump,
collects each box's `rationale` sentence, and embeds it with the frozen
SigLIP-SO400M text tower (pooled output, 1152-d). One pass, cached forever —
same "cache once, never regenerate" discipline as the Qwen outputs themselves.

Output: cache/rationale_<split>.pkl
  {"model": ..., "emb": {frame_key: {box_id: float16[1152]}}}
Boxes with no parse or an empty rationale are simply absent; the dataset feeds
zeros + a has-rationale flag of 0 for those.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u embed_rationale.py [--split val] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import config as C


def _qwen_cache_path(key: str) -> Path:
    vname, fid = key.rsplit("/", 1)
    return C.QWEN_CACHE_DIR / vname / f"{int(fid):05d}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="First N frames (debug). 0 = all.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    det_path = C.CACHE_DIR / f"detections_{args.split}.pkl"
    out_path = Path(args.out) if args.out else C.CACHE_DIR / f"rationale_{args.split}.pkl"
    print(f"[rat] loading detections: {det_path}", flush=True)
    with open(det_path, "rb") as f:
        keys = sorted(pickle.load(f)["records"].keys())
    if args.limit:
        keys = keys[: args.limit]
    print(f"[rat] {len(keys):,} frames", flush=True)

    # Gather (key, box_id, sentence) triples from the Qwen cache.
    triples: list[tuple[str, int, str]] = []
    n_missing = 0
    for key in keys:
        cpath = _qwen_cache_path(key)
        if not cpath.exists():
            n_missing += 1
            continue
        try:
            parsed = json.loads(cpath.read_text()).get("parsed") or []
        except Exception:
            n_missing += 1
            continue
        for bid, p in enumerate(parsed):
            if p and isinstance(p.get("rationale"), str) and p["rationale"].strip():
                triples.append((key, bid, p["rationale"].strip()))
    print(f"[rat] {len(triples):,} rationales (missing-qwen frames: {n_missing})", flush=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[rat] loading {C.RATIONALE_ENCODER} text tower on {device} ...", flush=True)
    from transformers import AutoTokenizer, SiglipTextModel

    tok = AutoTokenizer.from_pretrained(C.RATIONALE_ENCODER)
    enc = SiglipTextModel.from_pretrained(
        C.RATIONALE_ENCODER, torch_dtype=torch.float32
    ).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    emb: dict[str, dict[int, np.ndarray]] = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(triples), C.RATIONALE_BATCH):
            chunk = triples[s : s + C.RATIONALE_BATCH]
            # SigLIP text was trained with padding="max_length" — required for
            # faithful embeddings, not an optimization choice.
            inputs = tok([c[2] for c in chunk], padding="max_length",
                         max_length=C.RATIONALE_MAX_LEN, truncation=True,
                         return_tensors="pt").to(device)
            pooled = enc(**inputs).pooler_output                    # [b, 1152]
            pooled = pooled.cpu().to(torch.float16).numpy()
            for (key, bid, _), v in zip(chunk, pooled):
                emb.setdefault(key, {})[bid] = v
            done = s + len(chunk)
            if (done // C.RATIONALE_BATCH) % 20 == 0 or done == len(triples):
                el = time.time() - t0
                print(f"[rat] {done:,}/{len(triples):,}  {el:.0f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": C.RATIONALE_ENCODER, "dim": C.RATIONALE_DIM,
                     "emb": emb}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[rat] wrote {sum(len(v) for v in emb.values()):,} embeddings "
          f"({len(emb):,} frames) → {out_path}", flush=True)


if __name__ == "__main__":
    main()
