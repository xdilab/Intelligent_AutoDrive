# BDD-X CSV Column Schema

Source: `/data/datasets/BDD-X/BDD-X-Annotations_v1.csv` header row (61 columns total)

---

## Structure

The CSV uses a **wide format**: one row per annotator submission per video.
Up to 15 action segments can be annotated per submission (slots 1–15).

**Total columns: 61** = 1 (video URL) + 15 × 4 (start, end, action, justification per slot)

---

## Column List

| Column | Type | Description |
|--------|------|-------------|
| `Input.Video` | str (URL) | S3 URL to BDD100K video (may be stale; requires BDD100K download) |
| `Answer.1start` | int (str) | Start time of action segment 1 (seconds) |
| `Answer.1end` | int (str) | End time of action segment 1 (seconds) |
| `Answer.1action` | str | Free-text: what the ego vehicle is doing |
| `Answer.1justification` | str | Free-text: why (causal explanation, usually "because...") |
| `Answer.2start` | int (str) | Start time of segment 2 |
| `Answer.2end` | int (str) | End time of segment 2 |
| `Answer.2action` | str | Action for segment 2 |
| `Answer.2justification` | str | Justification for segment 2 |
| ... | ... | (same pattern for slots 3–14) |
| `Answer.15start` | int (str) | Start time of segment 15 |
| `Answer.15end` | int (str) | End time of segment 15 |
| `Answer.15action` | str | Action for segment 15 |
| `Answer.15justification` | str | Justification for segment 15 |

Empty slots: `Answer.Naction` = `""`, `Answer.Nstart` = `""`, etc.

---

## Key Statistics (verified from CSV)

| Property | Value |
|----------|-------|
| Total rows | 12,997 |
| Unique videos | 7,000 |
| Average annotators per video | ~1.86 |
| Total annotated action segments | 26,538 |
| Average segments per video | ~3.8 |
| Mean segment duration | ~8.8 seconds |
| Mean justification length | 7.9 words |

Source: `wiki/datasets/bdd-x.md`

---

## Data Loading Notes

To parse all action/justification pairs from the CSV:

```python
import csv
pairs = []
with open('BDD-X-Annotations_v1.csv') as f:
    for row in csv.DictReader(f):
        for i in range(1, 16):
            action = row.get(f'Answer.{i}action', '').strip()
            justif = row.get(f'Answer.{i}justification', '').strip()
            if action:
                pairs.append({
                    'video': row['Input.Video'],
                    'start': row[f'Answer.{i}start'],
                    'end':   row[f'Answer.{i}end'],
                    'action': action,
                    'justification': justif,
                })
# Result: ~26,538 pairs
```
