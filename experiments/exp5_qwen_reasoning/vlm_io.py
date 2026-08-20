"""Prompt construction, box overlay, and Qwen-JSON parsing for exp5.

Kept free of heavy deps (no torch / transformers) so it can be unit-tested and
imported from both the inference and eval stages.
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# Distinct colors cycled across boxes so the numeric id on the image is legible.
_BOX_COLORS = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
]


def draw_numbered_boxes(
    img: Image.Image,
    boxes_xyxy: Sequence[Sequence[float]],
) -> Image.Image:
    """Draw each box with its 0-based index label. `boxes_xyxy` are in `img` pixel
    coords already (caller rescales detector boxes to the native frame first)."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(i)
        tx, ty = x1 + 2, max(0, y1 - 24)
        # Background chip behind the id for contrast.
        try:
            bb = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(bb, fill=color)
        except Exception:
            pass
        draw.text((tx, ty), label, fill="black", font=font)
    return out


def format_priors(
    logits_row,                        # [184] array-like, raw flat detector logits
    agent_labels: List[str],
    action_labels: List[str],
    loc_labels: List[str],
) -> Dict[str, Tuple[str, float]]:
    """Detection-steered priors for ONE box: the frozen detector's top-1 class
    per head (agent/action/location) with its sigmoid confidence, read off the
    flat 184-dim logits (layout: [agentness | agent 10 | action 22 | loc 16 |
    duplex 49 | triplet 86])."""
    def _sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-float(x)))

    out: Dict[str, Tuple[str, float]] = {}
    off = 1
    for head, labels in (("agent", agent_labels), ("action", action_labels),
                         ("location", loc_labels)):
        block = [float(v) for v in logits_row[off : off + len(labels)]]
        best = max(range(len(block)), key=block.__getitem__)
        out[head] = (labels[best], _sig(block[best]))
        off += len(labels)
    return out


def _priors_section(priors: Sequence[Dict[str, Tuple[str, float]]]) -> str:
    """Render per-box detector predictions as a prompt block."""
    lines = []
    for i, p in enumerate(priors):
        parts = [f"{head}={lab} ({conf:.2f})" for head, (lab, conf) in p.items()]
        lines.append(f"  {i}: " + ", ".join(parts))
    body = "\n".join(lines)
    return f"""
DETECTOR PREDICTIONS (priors):
A trained video detector produced these boxes and also predicted, per box, the most likely agent, action and location (confidence 0-1 in parentheses):
{body}

Treat these as priors, not answers: verify each against the image. Keep a prediction when the image is consistent with it; replace it when the image contradicts it (especially action and location, where scene context matters). Do not copy the priors blindly, and do not let a low-confidence prior stop you from choosing a better label.
"""


def build_prompt(
    n_boxes: int,
    agent_labels: List[str],
    action_labels: List[str],
    loc_labels: List[str],
    priors: Optional[Sequence[Dict[str, Tuple[str, float]]]] = None,
) -> str:
    """Structured, schema-constrained prompt. Qwen judges agent/action/location/
    risk/rationale per box; duplex & triplet are derived downstream (valid-by-
    construction from agent x action x location, matching how GT is built).

    With `priors` (one dict per box from format_priors), the prompt becomes
    detection-steered: the detector's per-box class predictions are shown and
    Qwen verifies/refines them instead of classifying from scratch."""
    agents = ", ".join(agent_labels)
    actions = ", ".join(action_labels)
    locs = ", ".join(loc_labels)
    steered = _priors_section(priors) if priors is not None else ""
    return f"""You are a driving-scene perception assistant. The image is a single frame from a Waymo autonomous-driving video. {n_boxes} candidate objects have been detected and drawn as numbered colored boxes (ids 0..{n_boxes - 1}).
{steered}
For EACH numbered box, classify the object it encloses using ONLY the label vocabularies below. Use the exact label strings as written.

AGENT (choose exactly one — the object category):
  {agents}

ACTION (choose one or more — what the object is doing):
  {actions}

LOCATION (choose zero or more — where it is in the road scene):
  {locs}

Also assess:
  risk: one of [low, medium, high] — collision/interaction risk to the ego vehicle.
  rationale: one short sentence justifying the action and risk.

Rules:
- Output STRICT JSON only. No prose, no markdown fences.
- One JSON object per box, in a single JSON array, ordered by box id.
- Every box id 0..{n_boxes - 1} must appear exactly once.
- agent must be a single string from the AGENT list.
- actions and locations must be arrays of strings from their lists (may be empty for locations).

Output schema (array of {n_boxes} objects):
[
  {{"box_id": 0, "agent": "<agent>", "actions": ["<action>", ...], "locations": ["<loc>", ...], "risk": "<low|medium|high>", "rationale": "<one sentence>"}},
  ...
]"""


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
def _extract_json_array(text: str) -> Optional[list]:
    """Pull the first top-level JSON array out of a model response, tolerating
    markdown fences and trailing prose."""
    t = text.strip()
    # Strip ```json ... ``` fences if present. Tolerate a missing *closing* fence:
    # when generation hits the token cap mid-array there is an opening ```json but
    # no close, so fall back to stripping just the opening fence.
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    else:
        open_fence = re.search(r"```(?:json)?\s*(.*)", t, re.DOTALL)
        if open_fence:
            t = open_fence.group(1).strip()
    start = t.find("[")
    if start == -1:
        return None
    # Walk braces to find the matching close bracket.
    depth = 0
    for i in range(start, len(t)):
        c = t[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                chunk = t[start : i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    break
    # Truncated or malformed array: salvage every complete object by closing the
    # array at the last '}'. Boxes are emitted in detector-score order, so the
    # recovered prefix is the highest-confidence subset.
    last = t.rfind("}")
    if last > start:
        try:
            return json.loads(t[start : last + 1] + "]")
        except json.JSONDecodeError:
            return None
    return None


def _salvage_boxes(
    text: str,
    n_boxes: int,
    agent_set: set,
    action_set: set,
    loc_set: set,
) -> Dict[int, dict]:
    """Per-object regex salvage for structurally corrupted arrays (e.g. fused
    objects missing the `}, {` boundary — seen from the exp8 joint-tuned model,
    which sometimes blends its two training JSON dialects). Splits the raw text
    at each `"box_id": N` and reads that box's fields from its own segment.
    Faithful recovery only: fields are extracted verbatim and validated against
    the label sets exactly like the strict path; nothing is inferred."""
    hits = list(re.finditer(r'"box_id"\s*:\s*(\d+)', text))
    out: Dict[int, dict] = {}
    for j, m in enumerate(hits):
        bid = int(m.group(1))
        if not (0 <= bid < n_boxes) or bid in out:
            continue
        seg = text[m.end(): hits[j + 1].start() if j + 1 < len(hits) else len(text)]
        agent_m = re.search(r'"agent"\s*:\s*(?:null|"([^"]*)")', seg)
        risk_m = re.search(r'"risk"\s*:\s*"(low|medium|high)"', seg)
        rat_m = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', seg)

        def _strs(field: str) -> list:
            lm = re.search(r'"%s"\s*:\s*\[([^\]]*)\]' % field, seg)
            return re.findall(r'"([^"]+)"', lm.group(1)) if lm else []

        agent = agent_m.group(1) if agent_m else None
        out[bid] = {
            "agent": agent if agent in agent_set else None,
            "actions": [a for a in _strs("actions") if a in action_set],
            "locations": [l for l in _strs("locations") if l in loc_set],
            "risk": risk_m.group(1) if risk_m else None,
            "rationale": rat_m.group(1) if rat_m else None,
        }
    return out


def parse_qwen_response(
    text: str,
    n_boxes: int,
    agent_set: set,
    action_set: set,
    loc_set: set,
) -> List[Optional[dict]]:
    """Return a length-`n_boxes` list. Each entry is a normalized dict
    {agent, actions, locations, risk, rationale} or None if the box was not
    parseable. Invalid labels are dropped (kept faithful: no guessing).
    Boxes the strict array parse cannot recover fall back to per-object
    salvage (_salvage_boxes); strict results always win."""
    out: List[Optional[dict]] = [None] * n_boxes
    arr = _extract_json_array(text)
    if not isinstance(arr, list):
        arr = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        bid = obj.get("box_id")
        if not isinstance(bid, int) or not (0 <= bid < n_boxes):
            continue
        agent = obj.get("agent")
        agent = agent if agent in agent_set else None
        actions = [a for a in (obj.get("actions") or []) if a in action_set]
        locations = [l for l in (obj.get("locations") or []) if l in loc_set]
        risk = obj.get("risk")
        risk = risk if risk in ("low", "medium", "high") else None
        out[bid] = {
            "agent": agent,
            "actions": actions,
            "locations": locations,
            "risk": risk,
            "rationale": obj.get("rationale") if isinstance(obj.get("rationale"), str) else None,
        }
    if any(p is None for p in out):
        for bid, sal in _salvage_boxes(text, n_boxes, agent_set, action_set, loc_set).items():
            if out[bid] is None:
                out[bid] = sal
    return out
