#!/usr/bin/env python3
"""One-shot status collection across all active research tracks.

Read-only. Prints a compact report for periodic session review (see AGENTS.md
periodic log-inspection rule). Does not launch, repair, or modify anything.
"""
import glob, json, os, subprocess, sys, time

WIKI = '/data/repos/wiki'
MON = f'{WIKI}/artifacts/research-monitor'
def discover_local():
    """Any artifact dir carrying a pipeline-progress.json is an active local study.

    Hardcoding the list meant stage7-stage5-20260918 went untracked for an hour
    on 2026-09-18; discover instead so new studies are picked up automatically.
    """
    found = []
    for d in sorted(glob.glob(f'{WIKI}/artifacts/*')):
        if os.path.isfile(f'{d}/pipeline-progress.json'):
            found.append(d)
    return found
CLUSTER_ROOT = '/work/bbyrd1/stage56-full-20260914'


def jload(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def age(ts):
    return f'{(time.time() - ts) / 60:.1f}m'


def watchdog():
    print('== NCShare watchdog ==')
    hb = jload(f'{MON}/heartbeat.json')
    if not hb:
        print('  heartbeat: MISSING')
        return
    print(f'  heartbeat age: {age(hb["checked_unix"])} (checked {hb["checked_utc"]})')
    running = [j for j in hb.get('jobs', []) if j['status'] == 'RUNNING']
    pending = [j for j in hb.get('jobs', []) if j['status'] == 'PENDING']
    for j in running:
        print(f'  RUNNING {j["job"]:<12} {j["name"]:<22} {j["elapsed"]:>12}  {j["reason"]}')
    print(f'  PENDING: {len(pending)} jobs')
    try:
        with open(f'{MON}/alerts.jsonl') as fh:
            alerts = [json.loads(l) for l in fh if l.strip()]
        for a in alerts[-3:]:
            print(f'  ALERT [{a["severity"]}] {a["time"]} {a["message"]}')
    except Exception:
        pass


def cluster_progress():
    print('== NCShare job progress (live tail) ==')
    cmd = (
        f'R={CLUSTER_ROOT}; '
        'for f in 0 1 2 3 4 5; do printf "  train-%s: " $f; '
        'tail -1 $R/results/full56-train-732170-$f.log 2>/dev/null; done; '
        'for f in 1 2; do printf "  epoch1-%s: " $f; '
        'tail -1 $R/results/full56-epoch1-733585-$f.log 2>/dev/null; done'
    )
    try:
        out = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'ncshare', cmd],
            capture_output=True, text=True, timeout=90)
        print(out.stdout.rstrip() or '  (no output)')
        if out.returncode:
            print(f'  ssh exit {out.returncode}: {out.stderr.strip()[:200]}')
    except Exception as exc:
        print(f'  ssh failed: {exc}')


def local_study(root):
    name = os.path.basename(root)
    print(f'== {name} ==')
    proto = jload(f'{root}/protocol.json', {})
    expected = len(proto.get('conditions', [])) * len(proto.get('seeds', []))
    prog = jload(f'{root}/pipeline-progress.json', {})
    print(f'  phase: {prog.get("phase")}  pid {prog.get("pid")}  ({age(prog.get("time", time.time()))} ago)')
    if prog.get('dependency'):
        dep = prog['dependency']
        print(f'  waiting on: {dep} [{"present" if os.path.exists(dep) else "absent"}]')
    runs = f'{root}/runs'
    done = started = 0
    if os.path.isdir(runs):
        for d in sorted(os.listdir(runs)):
            started += 1
            if os.path.exists(f'{runs}/{d}/detector-results.json'):
                done += 1
    print(f'  runs: {done} with detector results / {started} started / {expected or "?"} planned')
    # live processes
    try:
        ps = subprocess.run(['ps', '-eo', 'pid,etime,pcpu,cmd'], capture_output=True, text=True, timeout=20).stdout
        live = [l for l in ps.splitlines() if root in l and 'grep' not in l
                and ('train_cached' in l or 'evaluate_cached' in l or 'pipeline' in l)]
        for l in live:
            f = l.split()
            print(f'  live pid {f[0]:<8} {f[1]:>10} cpu{f[2]:>6}  {f[-1].split("/")[-1] if "--" not in l else " ".join(f[4:])[:90]}')
    except Exception:
        pass
    try:
        with open(f'{root}/alerts.jsonl') as fh:
            for a in [json.loads(l) for l in fh if l.strip()][-2:]:
                print(f'  ALERT {a.get("message")}')
    except Exception:
        pass


def repair():
    print('== autonomous repair ==')
    try:
        out = subprocess.run(
            [sys.executable, '/data/repos/ROAD_Reason/research/post-proposal/monitoring/repair_dispatch.py', '--inspect'],
            capture_output=True, text=True, timeout=120)
        body = out.stdout.strip()
        print(f'  open incidents: {body if body else "(none)"}')
    except Exception as exc:
        print(f'  inspect failed: {exc}')


def gpus():
    print('== local GPUs ==')
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        for l in out.stdout.strip().splitlines():
            print(f'  {l}')
    except Exception as exc:
        print(f'  nvidia-smi failed: {exc}')


if __name__ == '__main__':
    print(f'### status {time.strftime("%Y-%m-%d %H:%M:%S %Z")}')
    watchdog()
    if '--no-ssh' not in sys.argv:
        cluster_progress()
    roots = discover_local()
    print(f'== local studies discovered: {len(roots)} ==')
    for r in roots:
        local_study(r)
    gpus()
    repair()
