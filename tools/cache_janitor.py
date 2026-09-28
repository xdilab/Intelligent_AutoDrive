"""LRU-style cache janitor for experiment feature caches.

Inventories large artifacts (*.pkl, *.pt > 300MB) under /data/repos/ROAD_Reason,
ranks by last access (LRU first), classifies regenerability by pattern, and
evicts only on explicit request. Every eviction appends the file, its size, and
its regeneration command to MANIFEST so nothing is lost silently.

Usage:
  python3 cache_janitor.py --report            # ranked table, no changes
  python3 cache_janitor.py --evict-gb 40       # delete LRU regenerable files
                                               # until ~40 GB freed (asks per file)
  python3 cache_janitor.py --evict-gb 40 --yes # no per-file prompt
"""
import argparse, os, sys, time
from pathlib import Path

ROOT = Path("/data/repos/ROAD_Reason")
MANIFEST = ROOT / "tools" / "evicted_manifest.md"
MIN_BYTES = 300 * 1024 * 1024

# pattern -> (regenerable, how)
RULES = [
    ("clip_feats_", True, "exp12_phrase_head/cache_clip_feats.py --split {train,val} [--clips]"),
    ("crop_feats_", True, "exp12_phrase_head/cache_crop_feats.py --split {train,val} (or NCShare /work copy)"),
    ("roi_feats_", True, "exp11_yolo/cache_roi_feats.py --split {train,val} [--junk]"),
    ("baseline_compat_dets", True, "exp2* eval scripts regenerate from baseline ckpt"),
    ("detections_", True, "exp6_detection_steered cache scripts"),
    ("dets_", False, "KEEP: YOLO/I3D full-candidate dumps are protocol row definitions"),
    (".pt", False, "KEEP: checkpoints are results, not caches"),
]

def classify(p):
    n = p.name
    for pat, regen, how in RULES:
        if pat in n:
            return regen, how
    return False, "unclassified: keep"

def inventory():
    out = []
    for p in ROOT.rglob("*"):
        if p.suffix not in (".pkl", ".pt") or not p.is_file():
            continue
        st = p.stat()
        if st.st_size < MIN_BYTES:
            continue
        regen, how = classify(p)
        out.append((st.st_atime, st.st_size, p, regen, how))
    out.sort()  # oldest access first = LRU first
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--report", action="store_true")
ap.add_argument("--evict-gb", type=float, default=0)
ap.add_argument("--yes", action="store_true")
args = ap.parse_args()

inv = inventory()
total = sum(s for _, s, *_ in inv)
print(f"{len(inv)} artifacts, {total/1e9:.1f} GB total\n")
now = time.time()
for at, sz, p, regen, how in inv:
    age = (now - at) / 86400
    tag = "EVICTABLE" if regen else "keep     "
    print(f"{tag} {sz/1e9:6.1f}G  atime {age:5.1f}d  {p.relative_to(ROOT)}")

if args.evict_gb > 0:
    freed = 0.0
    with open(MANIFEST, "a") as mf:
        for at, sz, p, regen, how in inv:
            if freed >= args.evict_gb * 1e9:
                break
            if not regen:
                continue
            if not args.yes:
                r = input(f"delete {p} ({sz/1e9:.1f}G)? [y/N] ")
                if r.lower() != "y":
                    continue
            mf.write(f"- {time.strftime('%Y-%m-%d')} deleted `{p}` ({sz/1e9:.1f}G); regenerate: {how}\n")
            p.unlink()
            freed += sz
            print(f"deleted {p} ({sz/1e9:.1f}G)")
    print(f"freed {freed/1e9:.1f} GB; manifest: {MANIFEST}")
