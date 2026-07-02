"""Re-parse cached Qwen responses in place from their stored `raw` text.

The first run cached `raw` correctly but populated `parsed` with a buggy
extractor that dropped truncated / fenced arrays (all-null). This rewrites the
`parsed` field using the current vlm_io.parse_qwen_response — no GPU, no
re-generation. Safe to run while qwen_infer is still writing new frames; it only
rewrites the `parsed` array of files that already exist.

  python -u reparse_cache.py [--split val]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import config as C
import vlm_io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    with open(C.CACHE_DIR / f"detections_{args.split}.pkl", "rb") as f:
        labels = pickle.load(f)["labels"]
    agent_set = set(labels["agent"])
    action_set = set(labels["action"])
    loc_set = set(labels["loc"])

    files = sorted(C.QWEN_CACHE_DIR.glob("*/*.json"))
    n_files = n_changed = n_boxes_before = n_boxes_after = 0
    for fp in files:
        try:
            qc = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # mid-write from the live job; skip this pass
        if qc.get("skipped") or not qc.get("raw"):
            continue
        n_files += 1
        before = sum(1 for p in (qc.get("parsed") or []) if p)
        parsed = vlm_io.parse_qwen_response(
            qc["raw"], int(qc["n_boxes"]), agent_set, action_set, loc_set
        )
        after = sum(1 for p in parsed if p)
        n_boxes_before += before
        n_boxes_after += after
        if after != before:
            qc["parsed"] = parsed
            fp.write_text(json.dumps(qc))
            n_changed += 1

    print(f"[reparse] files={n_files} changed={n_changed} "
          f"parsed-boxes {n_boxes_before} -> {n_boxes_after}", flush=True)


if __name__ == "__main__":
    main()
