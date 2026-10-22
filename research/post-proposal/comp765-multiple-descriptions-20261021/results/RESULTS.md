# COMP 765 detector results

COMP 765 — Multiple Class Descriptions for Long-Tail Road-Event Recognition

All values are detector frame AP@0.5 percentages; deltas are percentage points. Fresh paired seeds0,1,2.

| Endpoint | Single mean ± SD | Multi mean ± SD | Paired Δ | 95% paired-t CI |
|---|---:|---:|---:|---:|
| tail47 | 5.7952 ± 0.1040 | 5.6608 ± 0.0883 | -0.1344 | [-0.5006, +0.2317] |
| triplet | 9.0231 ± 0.1729 | 8.6336 ± 0.1568 | -0.3895 | [-1.1161, +0.3371] |
| deep28 | 4.6429 ± 0.0774 | 4.6228 ± 0.1113 | -0.0201 | [-0.4566, +0.4163] |
| common39 | 12.9132 ± 0.2565 | 12.2162 ± 0.2493 | -0.6969 | [-1.8611, +0.4673] |

| Seed | Single Tail47 | Multi Tail47 | Δ Tail47 |
|---|---:|---:|---:|
| 0 | 5.6963 | 5.6181 | -0.0782 |
| 1 | 5.7858 | 5.7623 | -0.0235 |
| 2 | 5.9036 | 5.6019 | -0.3017 |

The primary paired interval crosses zero: this three-seed study is inconclusive about a reliable Tail47 improvement. This is not evidence of equivalence.

Tail classes by mean paired AP change: 18 improve, 28 decline, 1 tie. All86 classes, all per-seed differences and all standard group metrics are exported; no success-only selection.

With only three training seeds the paired-t interval is fragile and does not quantify uncertainty over new datasets. Secondary endpoints and per-class differences are descriptive, not multiplicity-corrected confirmatory tests.

Averaging changes prototype geometry as well as the wording representation. This practical comparison does not isolate linguistic semantics or distinguish averaging from the quality of added descriptions. Canonical wording imprecisions are retained and disclosed in the description manifest.

No thesis records were modified. See protocol, freeze, input hashes, dependency versions, exact source commit and checkpoint-selection records for reproducibility.
