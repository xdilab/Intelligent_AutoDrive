# Expanded-data candidate — October 19, 2026

User authorized expanding training data while the real-time v1 student remains available. No default-model promotion is automatic.

Data-only change:64 expert-training videos,12 selection videos,12 student-held-out development videos;8 consecutive annotated frames each,704 total. Fixed hash ranking and earliest annotated window at/after40 remain unchanged. Original8 training,3 selection and3 held-out videos retain their roles. Development ranking0:3 plus6:15 selects checkpoints; ranking3:6 plus15:24 is held out. Gate and official validation stay untouched. Reuse112 verified pilot frames; generate592 new frames.

Keep the frozen shared ResNet50-layer3 backbone, causal scene GRU256, losses, learning rate, seed0,30epochs and supervised/distilled comparison. Select checkpoints by selection BCE. Report both expanded holdout and the original550-proposal holdout so the data change has a direct comparison. Proposal metrics remain separate from official detector AP. This is not backbone fine-tuning.

Pipeline: resumable per-video teacher/feature preparation → paired training → paired latency → fresh runtime replay. Completed cache hashes and input JPEG hashes are retained. Source/config changes fail resume checks. Each stage has a timeout; service runtime capped at2hours. Outputs/checkpoints are isolated from the pilot and real-time v1 bundle.

Initial ETA: first expanded checkpoint25–40min; full comparison30–45min. Basis:592 new frames × roughly2.35s/frame observed in the pilot, plus~40s loading/annotation and allowances for different candidate density, preparation I/O and evaluation. No queue; local GPU0. Head training and larger latency test provisionally3–8min. Refresh from progress JSON; these are proposal results, with no detector-AP evaluation scheduled.

Local units: `research-cseb-student-expanded.service` and `research-cseb-student-monitor.timer`. The timer inspects actual logs/progress and checkpoint contents every2minutes, creates hash-verified local backups, and repeats unresolved alerts with escalation metadata. Alerts are silent JSON/log records; desktop banners stay disabled. The monitor does not promise unattended agent repair. Workstation sleep/power loss interrupts execution. No NCShare job is submitted; local unit/PIDs/root are registered in wiki research-monitor/local-studies/.

## Completion verified October 21

The expanded run finished successfully on October 19. Its 96-frame, 12-video student-held-out development evaluation covers 45 of 86 triplet classes: supervised **17.674%**, distilled **21.028%**, fixed teacher **41.224%** proposal triplet AP. Selected epochs were 8 and 20. These are proposal metrics, not official detector AP, and this pool belongs to the teacher's historical development set.

Actual paired latency records contain 48 frames (four per video), with the first frame of each video excluded from the medians: student 39.10 ms, teacher 1.933 s, a 49.45× ratio. The old `latency.json` scope string incorrectly says 12 frames; the immutable records remain the authority. Fresh eight-frame runtime replay matched exactly. No promotion to the working realtime-v1 bundle occurred.

Verified artifacts: `/home/brandon/.local/share/road-reason/cseb-student-expanded-20261019/`, including `results.json`, `latency.json`, `runtime-verification.json`, logs and checkpoint backups.

The new [COMP 765 controlled study](../comp765-cseb-student-20261021/README.md) retains this architecture and loss, extends sequences, uses three paired seeds, and adds official detector evaluation and a raw-video runtime audit. Its artifacts are separate from this historical expansion.
