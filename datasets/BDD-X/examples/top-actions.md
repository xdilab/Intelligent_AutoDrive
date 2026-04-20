# BDD-X: Top Action Verbs

Source: `wiki/datasets/bdd-x.md` (verified from BDD-X-Annotations_v1.csv)

---

## Most Frequent Action Verbs (by occurrence)

| Verb | Count | Example action string |
|------|-------|-----------------------|
| slows | 2,632 | "The car slows down" |
| stopped | 2,520 | "The car has stopped" |
| accelerates | 1,988 | "The car accelerates" |
| moving | ~1,800 | "The car is moving at a steady speed" |
| driving | ~1,600 | "The car is driving straight ahead" |
| turns | ~1,400 | "The car turns left" |
| stops | ~1,200 | "The car stops" |
| moves | ~900 | "The car moves forward" |

---

## Notes

- Action strings are **free text** — no controlled vocabulary or taxonomy
- Justifications typically begin with "because..."
- Annotators are driving instructors (US traffic context)
- Mean justification: 7.9 words (compact causal statements)
- Mean action duration: ~8.8 seconds (covers behavior change events)

---

## Example Pairs (from CSV row 1, video 06d501fd-a9ffc960)

```
"The car accelerates"           → "because the light has turned green."
"The car is moving at a steady speed" → "because traffic is clear."
"The car slows slightly"        → "because it's turning into the right lane."
"The car stops"                 → "because it turns to the right."
"The car accelerates"           → "because traffic is clear."
```

---

## Contrast with CoVLA

BDD-X actions are human-authored and causal.
CoVLA plain_captions are auto-generated via rule-based + VideoLLaMA2-7B.
Both serve as Y/W supervision, but at different quality/scale tradeoffs:

| | BDD-X | CoVLA |
|-|-------|-------|
| Instances | 26,538 | 6,000,000 |
| Annotation | Manual (instructors) | Auto (rule-based + VLM) |
| Granularity | Action-level (~8.8s) | Frame-level (20 FPS) |
| Reasoning quality | High (explicit causal) | Medium (VLM-generated) |
