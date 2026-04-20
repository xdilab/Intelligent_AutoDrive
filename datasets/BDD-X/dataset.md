# BDD-X Dataset Reference

**For:** Dr. Moradi meeting — implementation-level detail  
**Sources:** `/data/datasets/BDD-X/BDD-X-Annotations_v1.csv`, `wiki/datasets/bdd-x.md`, `wiki/datasets/bdd100k.md`  
**Paper:** Kim et al., "Textual Explanations for Self-Driving Vehicles," ECCV 2018

---

## Dataset Summary

| Property | Value | Source |
|----------|-------|--------|
| Unique videos | 7,000 | Verified from CSV |
| CSV rows (annotator submissions) | 12,997 | Verified from CSV |
| Avg annotators per video | ~1.86 | 12,997 / 7,000 |
| Total annotated action segments | 26,538 | Verified from CSV |
| Avg actions per video | ~3.8 | 26,538 / 7,000 |
| Mean segment duration | ~8.8 seconds | Verified from CSV |
| Mean justification length | 7.9 words | Verified from CSV |
| Total frames | ~8.4M | 7,000 videos × ~40s × 30 FPS |
| Total hours | 77+ | Verified |
| Train split | 5,588 videos | `train.txt` |
| Val split | 698 videos | `val.txt` |
| Test split | 698 videos | `test.txt` |
| Video source | BDD100K dashcam videos | Kim et al., ECCV 2018 |
| Geographic region | USA (diverse) | Paper |

---

## What BDD-X Is

BDD-X is an **explanation overlay** on top of BDD100K videos. Annotators (driving instructors) watched BDD100K dashcam footage and wrote:
1. **Action:** What the ego vehicle is doing during a time segment
2. **Justification:** Why the ego vehicle is doing it (causal explanation)

BDD-X does **not** add new detection classes or bounding boxes. It adds behavioral explanation text over existing video footage. The 10 BDD100K detection classes (Pedestrian, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Traffic Light, Traffic Sign) remain unchanged.

---

## BDD100K Base Dataset (Detection Classes)

| Index | Class |
|-------|-------|
| 0 | Pedestrian |
| 1 | Rider |
| 2 | Car |
| 3 | Truck |
| 4 | Bus |
| 5 | Train |
| 6 | Motorcycle |
| 7 | Bicycle |
| 8 | Traffic Light |
| 9 | Traffic Sign |

Source: `wiki/datasets/bdd100k.md`, verified against BDD100K detection benchmark.

---

## CSV Annotation Schema

File: `/data/datasets/BDD-X/BDD-X-Annotations_v1.csv` (3.3 MB, 12,997 rows + header)

**Format: wide CSV, 61 columns**
- 1 column: `Input.Video` (S3 URL to BDD100K video)
- 60 columns: 15 action slots × 4 fields each (start, end, action, justification)

### Column List

| Column | Type | Description |
|--------|------|-------------|
| `Input.Video` | str (URL) | S3 URL to BDD100K .mov file (may be stale; need separate BDD100K download) |
| `Answer.1start` | int (as str) | Start time of action segment 1 (seconds) |
| `Answer.1end` | int (as str) | End time of action segment 1 (seconds) |
| `Answer.1action` | str | Free-text: what the ego vehicle is doing |
| `Answer.1justification` | str | Free-text: why (causal, usually "because...") |
| `Answer.2start` | int (as str) | Start time of segment 2 |
| `Answer.2end` | int (as str) | End time of segment 2 |
| `Answer.2action` | str | Action for segment 2 |
| `Answer.2justification` | str | Justification for segment 2 |
| ... | ... | Slots 3–15 follow the same pattern |
| `Answer.15start` | int (as str) | Start time of segment 15 |
| `Answer.15end` | int (as str) | End time of segment 15 |
| `Answer.15action` | str | Action for segment 15 |
| `Answer.15justification` | str | Justification for segment 15 |

Empty slots have empty strings for all four fields.

See `examples/column-schema.md` for full column list with parsing code.

---

## Real Examples (from CSV row 1: video `06d501fd-a9ffc960`)

| Slot | Start | End | Action | Justification |
|------|-------|-----|--------|---------------|
| 1 | 0s | 11s | The car accelerates | because the light has turned green. |
| 2 | 12s | 19s | The car is moving at a steady speed | because traffic is clear. |
| 3 | 20s | 22s | The car slows slightly | because it's turning into the right lane. |
| 4 | 23s | 36s | The car stops | because it turns to the right. |
| 5 | 37s | 40s | The car accelerates | because traffic is clear. |

See `examples/action-justification-pairs.md` and `examples/annotation-row.csv`.

---

## Annotation Process

Source: Kim et al., ECCV 2018

1. Annotators are **driving instructors** with knowledge of US traffic rules
2. Each annotator watches a BDD100K video (~40 seconds)
3. They identify **behavior change moments** (e.g., accelerate, slow down, stop, turn)
4. For each moment, they write:
   - **What:** Free-text action description (no controlled vocabulary)
   - **Why:** Causal explanation (encouraged to begin with "because")
5. They timestamp the start and end of each behavior
6. Up to 15 action slots per submission
7. Multiple annotators may submit for the same video (~1.86 avg)

---

## Split Files

Local paths: `/data/datasets/BDD-X/train.txt`, `val.txt`, `test.txt`  
Format: one video filename per line (without extension)

```
1_06d501fd-a9ffc960
2_01b0505f-5f564e84
3_06d501fd-fd237e38
4_06d54ae6-26a3446e
5_01b4e4b9-e21fe0a3
...
```

See `examples/split-sample.txt` for first 5 train entries.

---

## Top Action Verbs

| Verb | Count |
|------|-------|
| slows | 2,632 |
| stopped | 2,520 |
| accelerates | 1,988 |
| moving | ~1,800 |
| driving | ~1,600 |
| turns | ~1,400 |

Source: `wiki/datasets/bdd-x.md`

See `examples/top-actions.md` for full list with examples.

---

## Relevance to Approach 3 (Qwen2.5-VL Multi-Task)

| BDD-X field | DSDAG node | Supervision signal | Quality |
|-------------|------------|--------------------|---------|
| `Answer.Naction` | Y (action node) | What the ego vehicle is doing | Human-authored ✓✓ |
| `Answer.Njustification` | W (reason node) | Why the ego vehicle is doing it | Explicit causal, high quality ✓✓✓ |

BDD-X is the **highest-quality reasoning supervision** source for the W node.  
Justifications are human-authored by domain experts (instructors) — no VLM hallucination risk.  
Limitation: small scale (26,538 segments) vs. CoVLA (6M frames).

Training role: **Stage 1 pre-training** alongside CoVLA.  
Source: `wiki/methods/qwen25-vl-multitask.md`, `wiki/comparisons/bdd-x-vs-covla.md`

---

## Output Format (Model Output for BDD-X Task)

| Output | Format | Supervision | Metric |
|--------|--------|-------------|--------|
| Action | Language tokens | Cross-entropy vs. `Answer.Naction` | METEOR / CIDEr / BLEU |
| Justification | Language tokens | Cross-entropy vs. `Answer.Njustification` | METEOR / CIDEr / BLEU |

---

## Input to Model (Approach 3)

```
Input per segment:
  video:      BDD100K .mov clip, trimmed to [Answer.Nstart, Answer.Nend] seconds
  prompt:     "What is the vehicle doing and why?"
  supervision: action string (Y) + justification string (W)

Action-level granularity: one (action, justification) pair per ~8.8s segment
No frame-level annotations — segment is treated as a whole
```

---

## Data Flow

```
BDD-X-Annotations_v1.csv
  └─ parse rows → (video_url, start, end, action, justification) tuples
         │
         ▼
BDD100K video download (separate)
  └─ trim to [start, end] → video clip (~8.8s avg)
         │
         ▼
Qwen2.5-VL-7B video encoder (ViT + spatial merger)
  └─ Visual token sequence for trimmed clip
         │
         ▼
Language model head (autoregressive)
  └─ Generate action string → Generate justification string
         │
         ▼
L_bddx = cross-entropy(action) + cross-entropy(justification)
```

---

## BDD-X vs. CoVLA

| | BDD-X | CoVLA |
|-|-------|-------|
| Instances | 26,538 segments | 6,000,000 frames |
| Granularity | Action-level (~8.8s) | Frame-level (20 FPS) |
| Annotation | Manual (driving instructors) | Auto (rule-based + VideoLLaMA2-7B) |
| Reasoning quality | High — explicit causal "because" | Medium — VLM-generated |
| Domain | USA (diverse) | Tokyo, Japan |
| Y supervision | Free-text action | `plain_caption` |
| W supervision | Free-text justification | `risk` |
| No sensor data | ✓ (text only from video) | Full CAN bus + GPS + TL + lead vehicle |

Source: `wiki/comparisons/bdd-x-vs-covla.md`

---

## Local Paths

| Content | Path |
|---------|------|
| Annotations CSV | `/data/datasets/BDD-X/BDD-X-Annotations_v1.csv` |
| Train split | `/data/datasets/BDD-X/train.txt` |
| Val split | `/data/datasets/BDD-X/val.txt` |
| Test split | `/data/datasets/BDD-X/test.txt` |
| BDD100K videos | Separate download required (S3 URLs in CSV may be stale) |

---

## Examples in This Folder

| File | Contents |
|------|----------|
| `examples/annotation-row.csv` | Real CSV row (header + 1 annotator submission, 5 action slots) |
| `examples/action-justification-pairs.md` | 5 real (action, justification) pairs from first CSV row |
| `examples/split-sample.txt` | First 5 lines of train.txt |
| `examples/column-schema.md` | All 61 column names with descriptions + parsing code |
| `examples/top-actions.md` | Top action verbs with counts + CoVLA comparison |
