"""Rebuild road_waymo_yolo image symlinks on a new host from its label files."""
import os, sys
from pathlib import Path
root, frames = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for sp in ("train", "val"):
    (root / "images" / sp).mkdir(parents=True, exist_ok=True)
    for lbl in (root / "labels" / sp).iterdir():
        vname, fid = lbl.stem.rsplit("_", 1)
        src = frames / vname / f"{fid}.jpg"
        dst = root / "images" / sp / f"{lbl.stem}.jpg"
        assert src.exists(), src
        if not dst.exists():
            os.symlink(src, dst)
        n += 1
print(f"linked {n} images")
