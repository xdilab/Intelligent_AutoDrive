# Research monitoring

User requested proactive monitoring after a staging failure wasted hours (September15,2026).

`watch.py` runs as a one-CPU NCShare job, checks every2 minutes, persists heartbeat and deduplicated alerts. It watches current full-study job IDs, terminal failures, unsatisfiable dependencies, missing frame canaries after recovery verification, and30-minute no-log-progress warnings. Stall warnings are diagnostic, not proof of failure. No automatic model changes or arbitrary retries.

`pull.py` runs via a local systemd user timer every2 minutes, retrieves alerts/heartbeat, and sends desktop notifications. It warns on connection loss or a stale watchdog. Cluster checks continue while the workstation is off; desktop delivery waits until the workstation/user service is running again. This does not send messages into the Codex chat or guarantee automatic debugging. Three sequential seven-day allocations are registered in `/work/bbyrd1/research-monitor/renewals.json`: 732181 → 732183 → 732184 (`afterany`). This provides nominal coverage through October 6, subject to scheduler availability and early job failures; stale-heartbeat alerts cover watchdog loss/expiry. Check remaining coverage before each long launch.

Logs: cluster `/work/bbyrd1/research-monitor/`; local wiki `artifacts/research-monitor/`. For every future long experiment, register its job IDs/root with the watchdog and verify alert delivery before declaring it launched. Do not promise continuous manual observation or unattended repairs.

## Unattended research repair

`research-repair.timer` invokes `repair_dispatch.py` every two minutes. It reads
current monitor state and launches a separate `codex exec --approve-for-me`
session only for incidents. Read `repair-prompt.md` for the authorized scope.
The workspace sandbox and automatic approval reviewer remain enabled. The CLI
uses the existing authenticated account/configuration and consumes its usage.

Sessions are serialized with flock, capped at30 minutes, and retried no more
than once/hour and three times per incident. Exhaustion requires manual review;
notifications continue. The runner itself may not be edited by its repair agent.
Reports and complete event logs are under
`/data/repos/wiki/artifacts/research-repair/incident-/`.
Inspect the report and fresh study progress; exit0 is not a recovery guarantee.

Check: `systemctl --user status research-repair.timer`
Read-only collection: `python3 repair_dispatch.py --inspect`
Test: `python3 test_dispatch.py`
Stop future dispatch: `systemctl --user stop research-repair.timer`
Stop an active repair session: `systemctl --user stop research-repair.service`

The workstation must stay on/awake and connected. Linger is enabled for Brandon,
so user services survive logout. CLI authentication/usage and SSH must remain
available. Initial integration smoke verified a bounded file write and read-only
NCShare SSH; no actual research failure was injected. Notifications/report files
are the output channel, not guaranteed delivery into the interactive chat.
