# Research monitoring

User requested proactive monitoring after a staging failure wasted hours (September15,2026).

`watch.py` runs as a one-CPU NCShare job, checks every2 minutes, persists heartbeat and deduplicated alerts. It watches current full-study job IDs, terminal failures, unsatisfiable dependencies, missing frame canaries after recovery verification, and30-minute no-log-progress warnings. Stall warnings are diagnostic, not proof of failure. No automatic model changes or arbitrary retries.

`pull.py` runs via a local systemd user timer every2 minutes, retrieves alerts/heartbeat, and sends desktop notifications. It warns on connection loss or a stale watchdog. Cluster checks continue while the workstation is off; desktop delivery waits until the workstation/user service is running again. This does not send messages into the Codex chat or guarantee automatic debugging. Three sequential seven-day allocations are registered in `/work/bbyrd1/research-monitor/renewals.json`: 732181 → 732183 → 732184 (`afterany`). This provides nominal coverage through October 6, subject to scheduler availability and early job failures; stale-heartbeat alerts cover watchdog loss/expiry. Check remaining coverage before each long launch.

Logs: cluster `/work/bbyrd1/research-monitor/`; local wiki `artifacts/research-monitor/`. For every future long experiment, register its job IDs/root with the watchdog and verify alert delivery before declaring it launched. Do not promise continuous manual observation or unattended repairs.
