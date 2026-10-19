"""Export a bounded validation cache sample; stdlib only, safe on login nodes.

Read-only with respect to the study. No labels/targets are exported to inference.
"""
import argparse
import ast
import hashlib
import json
import math
import re
import shutil
import struct
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def slice_npy(source, target, start, stop):
    """Copy contiguous rows without importing NumPy or reading the entire array."""
    with source.open('rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('Invalid NPY magic')
        version = tuple(f.read(2))
        fmt = '<H' if version == (1, 0) else '<I'
        length = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        header = ast.literal_eval(f.read(length).decode('latin1'))
        if header['fortran_order'] or not re.fullmatch(r'[<>=|][fiu]\d+', header['descr']):
            raise ValueError('Only contiguous numeric NPY arrays supported')
        shape = header['shape']
        if not 0 <= start <= stop <= shape[0]:
            raise ValueError('Slice outside array')
        row_bytes = int(header['descr'][2:]) * math.prod(shape[1:])
        f.seek(start * row_bytes, 1)
        header['shape'] = (stop - start,) + shape[1:]
        encoded = repr(header).encode('latin1')
        encoded += b' ' * ((64 - (10 + len(encoded) + 1) % 64) % 64) + b'\n'
        with target.open('wb') as out:
            out.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(encoded)) + encoded)
            remaining = (stop - start) * row_bytes
            while remaining:
                block = f.read(min(1 << 20, remaining))
                if not block:
                    raise ValueError('Truncated NPY')
                out.write(block)
                remaining -= len(block)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--video', help='Default: first video in the locked validation manifest')
    ap.add_argument('--start-fid', type=int, default=0)
    a = ap.parse_args()
    if a.frames < 1 or a.frames > 600:
        ap.error('Use 1–600 frames per portable sample')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'cache').mkdir()
    (a.output / 'frames').mkdir()
    rows = []
    video = a.video
    for line in (a.root / 'data/val.jsonl').open():
        row = json.loads(line)
        video = video or row['video']
        if row['video'] != video or row['fid'] < a.start_fid:
            continue
        if rows and rows[-1]['row_end'] != row['row_start']:
            raise ValueError('Sample must be contiguous in official candidate order')
        rows.append(row)
        if len(rows) == a.frames:
            break
    if not rows:
        raise ValueError('No validation frames matched')
    start, stop = rows[0]['row_start'], rows[-1]['row_end']
    for name in ['q', 'agent']:
        slice_npy(a.root / f'data/val-{name}.npy', a.output / f'{name}.npy', start, stop)
    records = []
    for row in rows:
        key = f"{row['video']}_{row['fid']:05d}"
        if not re.fullmatch(r'[A-Za-z0-9_-]+', key):
            raise ValueError('Invalid frame key')
        cache = a.root / 'cache/val' / f'{key}.pt'
        frame = a.root / 'frames' / row['video'] / f"{row['fid']:05d}.jpg"
        frame_sha = sha(frame)
        if frame_sha != row['frame_sha256'][row['key_t']]:
            raise ValueError('Frame differs from encoder-cache input')
        shutil.copyfile(cache, a.output / 'cache' / cache.name)
        shutil.copyfile(frame, a.output / 'frames' / f'{key}.jpg')
        records.append({'key': key, 'video': row['video'], 'fid': row['fid'],
                        'row_start': row['row_start'] - start, 'row_end': row['row_end'] - start,
                        'cache_sha256': sha(cache), 'image_sha256': frame_sha,
                        'input_fids': row['fids'], 'key_t': row['key_t']})
    official = json.loads((a.root / 'data/shared-frames.json').read_text())
    result = {'format': 1, 'study_root': str(a.root), 'split': 'val',
              'selection': 'contiguous manifest frames, no prediction-based selection',
              'video': video, 'labels': official['labels'], 'frames': records,
              'source_row_start': start, 'source_row_end': stop,
              'cache_marker_sha256': sha(a.root / 'cache-ready.json'),
              'protocol_sha256': sha(a.root / 'protocol.json'),
              'q_sha256': sha(a.output / 'q.npy'), 'agent_sha256': sha(a.output / 'agent.npy')}
    (a.output / 'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'frames': len(rows), 'candidates': stop-start, 'output': str(a.output)}))


if __name__ == '__main__':
    main()
