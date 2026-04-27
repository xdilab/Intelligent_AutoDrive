"""
BDDXDataset — parses BDD-X-Annotations_v1.csv (wide format, up to 15 action slots)
and filters to examples whose images are present locally.

Split files (train.txt / val.txt / test.txt) use the format:
    {n}_{video_stem}     e.g. "1_06d501fd-a9ffc960"
Strip everything up to and including the first underscore to get the stem.
Images are at IMG_DIR/{stem}.jpg.

Each (image, action, justification) triple from a non-empty CSV slot is one example.
Multiple annotators per video and multiple slots per annotator are kept as separate examples.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


class BDDXDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        split_file: str,
        processor=None,          # unused here; kept for API consistency
        max_examples: Optional[int] = None,
    ):
        self.img_dir = img_dir
        self.examples: list[dict] = []

        # Load split stems: strip leading "{n}_" prefix
        with open(split_file) as f:
            split_stems = set(
                "_".join(line.strip().split("_")[1:])
                for line in f
                if line.strip()
            )

        # Parse wide CSV
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("Input.Video", "").strip()
                if not url:
                    continue
                stem = os.path.splitext(os.path.basename(url))[0]
                if stem not in split_stems:
                    continue
                img_path = os.path.join(img_dir, f"{stem}.jpg")
                if not os.path.exists(img_path):
                    continue
                for i in range(1, 16):
                    action = row.get(f"Answer.{i}action", "").strip()
                    justification = row.get(f"Answer.{i}justification", "").strip()
                    if action and justification:
                        self.examples.append(
                            {
                                "img_path": img_path,
                                "action": action,
                                "justification": justification,
                            }
                        )

        if max_examples is not None:
            self.examples = self.examples[:max_examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        image = Image.open(ex["img_path"]).convert("RGB")
        return {
            "image": image,
            "action": ex["action"],
            "justification": ex["justification"],
        }
