"""One requested desktop ping when all three setup checks pass; no repeat alerts."""
import argparse, datetime, fcntl, json, subprocess
from pathlib import Path

REMOTE = '/work/bbyrd1/contextual-roi-1b-20260930/results'
NAMES = ['assets.json', 'benchmark.json', 'text-benchmark.json']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(); args.state.mkdir(parents=True, exist_ok=True)
    with (args.state/'ping.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = args.state/'ping-attempted.json'
        if marker.exists():
            print('Already attempted the one-time completion ping.'); return
        code = ('import json;from pathlib import Path;'
                f'r=Path({REMOTE!r});names={NAMES!r};'
                'print(json.dumps({n:json.loads((r/n).read_text()) if (r/n).exists() else None for n in names}))')
        import shlex
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 'ncshare', 'python3 -c '+shlex.quote(code)],
                                capture_output=True, text=True, timeout=45, check=True)
        records = json.loads(result.stdout)
        ready = all(records.get(n) and records[n].get('passed') is True for n in NAMES)
        receipt = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'ready': ready, 'checks': records}
        tmp = args.state/'setup-status.tmp'; tmp.write_text(json.dumps(receipt, indent=2))
        tmp.replace(args.state/'setup-status.json')
        print('Setup compatibility complete:', ready)
        if not ready or args.dry_run: return
        # Persist before sending: no notification loop if the desktop call fails.
        marker.write_text(json.dumps({'time': receipt['time'], 'status': 'attempting'}, indent=2))
        sent = subprocess.run(['/usr/bin/notify-send', '--app-name=1B study', '--expire-time=15000',
                               'InternVideo2-1B compatibility checks finished',
                               'Weights verified; visual benchmark and all184 phrase encoding passed. '
                               'Full extraction/training is a separate next phase. Details are saved in the research wiki.'],
                              timeout=15, capture_output=True, text=True)
        marker.write_text(json.dumps({'time': receipt['time'], 'status': 'sent' if sent.returncode == 0 else 'failed',
                                      'returncode': sent.returncode, 'stderr': sent.stderr}, indent=2))


if __name__ == '__main__': main()
