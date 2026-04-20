# BDD-X: Real Action + Justification Pairs

Source: `/data/datasets/BDD-X/BDD-X-Annotations_v1.csv` — first annotator submission for video `06d501fd-a9ffc960`

---

## Video: `06d501fd-a9ffc960` (from BDD100K)

5 action segments annotated by a single annotator (driving instructor):

| Slot | Start (s) | End (s) | Action | Justification |
|------|-----------|---------|--------|---------------|
| 1 | 0 | 11 | The car accelerates | because the light has turned green. |
| 2 | 12 | 19 | The car is moving at a steady speed | because traffic is clear. |
| 3 | 20 | 22 | The car slows slightly | because it's turning into the right lane. |
| 4 | 23 | 36 | The car stops | because it turns to the right. |
| 5 | 37 | 40 | The car accelerates | because traffic is clear. |

---

## Annotation Format Notes

- **One CSV row = one annotator's submission for one video**
- A video may have multiple rows (up to 1.86 annotators on average)
- Each annotator annotates up to 15 action slots per submission (slots 1–15)
- Empty slots have empty strings for start/end/action/justification
- `Answer.Nstart` and `Answer.Nend` are in **seconds** (integer strings in CSV)

---

## DSDAG Mapping

| CSV field | DSDAG node | Role |
|-----------|-----------|------|
| `Answer.Naction` | Y (action node) | What the ego vehicle is doing |
| `Answer.Njustification` | W (reason node) | Why the ego vehicle is doing it |

---

## Annotation quality note

Annotators are **driving instructors** familiar with US traffic rules.
Justifications use explicit causal language ("because...").
This makes BDD-X the highest-quality reasoning supervision source for the W node,
despite being smaller than CoVLA (26,538 segments vs. 6M frames).

Reference: Kim et al., "Textual Explanations for Self-Driving Vehicles," ECCV 2018
