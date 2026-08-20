"""Clip-consistent DETR augmentations for video detection.

Adapted from Frozen-DETR/DINO-coco/datasets/transforms.py (Facebook Research).
All spatial transforms are sampled once per clip and applied identically to all
T frames. Box coordinates are adjusted per frame independently.

Our dataset returns normalized [x1,y1,x2,y2] in [0,1]. We convert to pixel
coordinates before augmentation and re-normalize after.

Train pipeline (matches DINO/Frozen-DETR with strong_aug):
    RandomHorizontalFlip(p=0.5)
    RandomSelect(
        RandomResize(scales, max_size=1333),          # path A
        Compose([                                      # path B
            RandomResize([400, 500, 600]),
            RandomSizeCrop(384, 600),
            RandomResize(scales, max_size=1333),
        ])
    )
    RandomSelect(AdjustBrightness, AdjustContrast, LightingNoise)  # strong aug

Val pipeline: identity (no augmentation).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import PIL.Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Standard DETR multi-scale sizes
DETR_SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
DETR_MAX_SIZE = 1333
DETR_SCALES2_RESIZE = [400, 500, 600]
DETR_SCALES2_CROP = (384, 600)


# ---------------------------------------------------------------------------
# Primitive ops (image + boxes in pixel coords [x1,y1,x2,y2])
# ---------------------------------------------------------------------------

def _hflip_image(image: PIL.Image.Image) -> PIL.Image.Image:
    return TF.hflip(image)


def _hflip_boxes(boxes: torch.Tensor, img_width: int) -> torch.Tensor:
    """Flip [x1,y1,x2,y2] boxes horizontally."""
    if boxes.numel() == 0:
        return boxes
    flipped = boxes.clone()
    flipped[:, 0] = img_width - boxes[:, 2]
    flipped[:, 2] = img_width - boxes[:, 0]
    return flipped


def _get_resize_size(
    image_size: Tuple[int, int], target_size: int, max_size: int | None = None
) -> Tuple[int, int]:
    """Compute (h, w) that resizes the shorter side to target_size, respecting max_size."""
    w, h = image_size
    if max_size is not None:
        min_orig = float(min(w, h))
        max_orig = float(max(w, h))
        if max_orig / min_orig * target_size > max_size:
            target_size = int(round(max_size * min_orig / max_orig))
    if w < h:
        ow = target_size
        oh = int(target_size * h / w)
    else:
        oh = target_size
        ow = int(target_size * w / h)
    return (oh, ow)


def _resize_image(image: PIL.Image.Image, size: Tuple[int, int]) -> PIL.Image.Image:
    return TF.resize(image, list(size), antialias=True)


def _resize_boxes(
    boxes: torch.Tensor,
    orig_size: Tuple[int, int],
    new_size: Tuple[int, int],
) -> torch.Tensor:
    """Scale boxes from orig_size (w,h) to new_size (h,w)."""
    if boxes.numel() == 0:
        return boxes
    orig_w, orig_h = orig_size
    new_h, new_w = new_size
    rw = new_w / orig_w
    rh = new_h / orig_h
    return boxes * torch.tensor([rw, rh, rw, rh], dtype=boxes.dtype)


def _crop_image(
    image: PIL.Image.Image, region: Tuple[int, int, int, int]
) -> PIL.Image.Image:
    return TF.crop(image, *region)


def _crop_boxes(
    boxes: torch.Tensor, region: Tuple[int, int, int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop boxes and return (cropped_boxes, keep_mask).

    region: (top, left, height, width)
    """
    if boxes.numel() == 0:
        return boxes, torch.ones(0, dtype=torch.bool)

    top, left, h, w = region
    cropped = boxes - torch.tensor([left, top, left, top], dtype=boxes.dtype)
    max_size = torch.tensor([w, h], dtype=boxes.dtype)
    cropped = torch.min(cropped.reshape(-1, 2, 2), max_size).reshape(-1, 4)
    cropped = cropped.clamp(min=0)

    # Keep boxes with positive area
    keep = (cropped[:, 2] > cropped[:, 0]) & (cropped[:, 3] > cropped[:, 1])
    return cropped, keep


# ---------------------------------------------------------------------------
# Normalized ↔ pixel coordinate conversion
# ---------------------------------------------------------------------------

def _denormalize_boxes(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """[0,1] normalized → pixel coordinates."""
    if boxes.numel() == 0:
        return boxes
    return boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)


def _normalize_boxes(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Pixel coordinates → [0,1] normalized."""
    if boxes.numel() == 0:
        return boxes
    return boxes / torch.tensor([w, h, w, h], dtype=boxes.dtype)


# ---------------------------------------------------------------------------
# Color augmentations (image-only, no box adjustment needed)
# Adapted from Frozen-DETR/DINO-coco/datasets/sltransform.py
# ---------------------------------------------------------------------------

def _adjust_brightness(image: PIL.Image.Image, max_factor: float = 2.0) -> PIL.Image.Image:
    """Random brightness: factor in [0.5*max, max]."""
    factor = ((random.random() + 1.0) / 2.0) * max_factor
    return TF.adjust_brightness(image, factor)


def _adjust_contrast(image: PIL.Image.Image, max_factor: float = 2.0) -> PIL.Image.Image:
    """Random contrast: factor in [0.5*max, max]."""
    factor = ((random.random() + 1.0) / 2.0) * max_factor
    return TF.adjust_contrast(image, factor)


def _lighting_noise(image: PIL.Image.Image) -> PIL.Image.Image:
    """Random RGB channel permutation."""
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms) - 1)]
    t = TF.to_tensor(image)
    t = t[list(swap), :, :]
    return TF.to_pil_image(t)


_COLOR_AUGMENTATIONS = [_adjust_brightness, _adjust_contrast, _lighting_noise]


# ---------------------------------------------------------------------------
# Clip-level augmentation
# ---------------------------------------------------------------------------

class ClipAugmentation:
    """Clip-consistent spatial augmentations for DETR-style training.

    Samples augmentation parameters once, applies the same spatial transform
    to all T frames, and adjusts per-frame boxes independently.

    Args:
        train: Whether to apply augmentation (False = identity for val).
        scales: Multi-scale resize targets.
        max_size: Maximum size for any dimension after resize.
        scales2_resize: Resize scales for the crop path.
        scales2_crop: (min_crop, max_crop) for RandomSizeCrop.
    """

    def __init__(
        self,
        train: bool = True,
        scales: list[int] | None = None,
        max_size: int = DETR_MAX_SIZE,
        scales2_resize: list[int] | None = None,
        scales2_crop: tuple[int, int] = DETR_SCALES2_CROP,
        strong_aug: bool = True,
    ):
        self.train = train
        self.scales = scales or DETR_SCALES
        self.max_size = max_size
        self.scales2_resize = scales2_resize or DETR_SCALES2_RESIZE
        self.scales2_crop = scales2_crop
        self.strong_aug = strong_aug

    def __call__(
        self,
        pil_frames: List[PIL.Image.Image],
        frame_targets: List[Optional[Dict[str, torch.Tensor]]],
    ) -> Tuple[List[PIL.Image.Image], List[Optional[Dict[str, torch.Tensor]]]]:
        if not self.train or len(pil_frames) == 0:
            return pil_frames, frame_targets

        # Get original image size (all frames in a clip have the same size)
        orig_w, orig_h = pil_frames[0].size

        # --- Sample augmentation parameters (once per clip) ---
        do_flip = random.random() < 0.5
        use_crop_path = random.random() < 0.5
        # Strong aug: pick one color transform to apply to all frames
        color_fn = random.choice(_COLOR_AUGMENTATIONS) if self.strong_aug else None

        if use_crop_path:
            # Path B: resize → crop → resize
            resize1_scale = random.choice(self.scales2_resize)
            resize1_size = _get_resize_size(
                (orig_w, orig_h), resize1_scale, max_size=None
            )
            # Crop params will be sampled after resize1 (need the resized size)
            final_scale = random.choice(self.scales)
        else:
            # Path A: just resize
            final_scale = random.choice(self.scales)

        # --- Apply to all frames ---
        aug_frames = []
        aug_targets = []

        # For crop path, we need the crop region to be the same for all frames.
        # Sample it from the first frame's resized dimensions.
        crop_region = None

        for i, (frame, target) in enumerate(zip(pil_frames, frame_targets)):
            cur_w, cur_h = frame.size

            # Convert boxes to pixel coords
            if target is not None and "boxes" in target:
                target = {k: v.clone() if isinstance(v, torch.Tensor) else v
                          for k, v in target.items()}
                target["boxes"] = _denormalize_boxes(target["boxes"], cur_w, cur_h)

            # Step 1: Horizontal flip
            if do_flip:
                frame = _hflip_image(frame)
                if target is not None and "boxes" in target:
                    target["boxes"] = _hflip_boxes(target["boxes"], cur_w)

            # Step 2: Resize (+ optional crop)
            if use_crop_path:
                # Resize to intermediate size
                frame = _resize_image(frame, resize1_size)
                if target is not None and "boxes" in target:
                    target["boxes"] = _resize_boxes(
                        target["boxes"], (cur_w, cur_h), resize1_size
                    )

                # Crop (sample region from first frame, reuse for all)
                resized_w, resized_h = frame.size
                if crop_region is None:
                    crop_w = random.randint(
                        self.scales2_crop[0],
                        min(resized_w, self.scales2_crop[1]),
                    )
                    crop_h = random.randint(
                        self.scales2_crop[0],
                        min(resized_h, self.scales2_crop[1]),
                    )
                    crop_region = T.RandomCrop.get_params(
                        frame, [crop_h, crop_w]
                    )

                frame = _crop_image(frame, crop_region)
                if target is not None and "boxes" in target:
                    target["boxes"], keep = _crop_boxes(
                        target["boxes"], crop_region
                    )
                    # Filter out zero-area boxes
                    for k, v in target.items():
                        if isinstance(v, torch.Tensor) and v.shape[0] == keep.shape[0]:
                            target[k] = v[keep]

                # Final resize
                post_crop_w, post_crop_h = frame.size
                final_size = _get_resize_size(
                    (post_crop_w, post_crop_h), final_scale, self.max_size
                )
                frame = _resize_image(frame, final_size)
                if target is not None and "boxes" in target:
                    target["boxes"] = _resize_boxes(
                        target["boxes"],
                        (post_crop_w, post_crop_h),
                        final_size,
                    )
            else:
                # Path A: single resize
                final_size = _get_resize_size(
                    (cur_w, cur_h), final_scale, self.max_size
                )
                frame = _resize_image(frame, final_size)
                if target is not None and "boxes" in target:
                    target["boxes"] = _resize_boxes(
                        target["boxes"], (cur_w, cur_h), final_size
                    )

            # Strong color augmentation (same transform for all frames)
            if color_fn is not None:
                frame = color_fn(frame)

            # Re-normalize boxes to [0,1]
            final_w, final_h = frame.size
            if target is not None and "boxes" in target:
                target["boxes"] = _normalize_boxes(
                    target["boxes"], final_w, final_h
                )

            # Handle frames that lost all boxes after crop
            if target is not None and "boxes" in target and target["boxes"].shape[0] == 0:
                target = None

            aug_frames.append(frame)
            aug_targets.append(target)

        return aug_frames, aug_targets
