"""
Script 1: count_combinations.py

Counts valid duplex (agent+action) and triplet (agent+action+location) combinations
in the ROAD-Waymo annotation file, then flags corner cases by frequency.

Usage:
    python count_combinations.py [--anno PATH] [--threshold N] [--pct PCT] [--out PATH]

Defaults:
    --anno      /data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json
    --threshold 50      (absolute count below this → corner case)
    --pct       15      (bottom N% by frequency also → corner case)
    --out       combination_counts.csv
"""

import json
import csv
import argparse
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--anno', default='/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json')
    p.add_argument('--threshold', type=int, default=50,
                   help='Absolute count below which a combo is a corner case')
    p.add_argument('--pct', type=float, default=15.0,
                   help='Bottom N%% by frequency also flagged as corner case')
    p.add_argument('--out', default='combination_counts.csv')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Loading {args.anno} ...')
    with open(args.anno) as f:
        data = json.load(f)

    all_duplex_labels  = data['all_duplex_labels']   # 152 possible
    all_triplet_labels = data['all_triplet_labels']  # 1620 possible
    valid_duplex_labels  = set(data['duplex_labels'])   # 49 valid
    valid_triplet_labels = set(data['triplet_labels'])  # 86 valid

    duplex_counts  = defaultdict(int)
    triplet_counts = defaultdict(int)

    db = data['db']
    n_videos = len(db)
    n_boxes  = 0

    for vi, (vid, vdata) in enumerate(db.items()):
        if vi % 100 == 0:
            print(f'  {vi}/{n_videos} videos processed ...')
        frames = vdata.get('frames', {})
        for fid, fdata in frames.items():
            if not fdata.get('annotated', 0):
                continue
            for anno in fdata.get('annos', {}).values():
                n_boxes += 1
                for did in anno.get('duplex_ids', []):
                    label = all_duplex_labels[did]
                    if label in valid_duplex_labels:
                        duplex_counts[label] += 1
                for tid in anno.get('triplet_ids', []):
                    label = all_triplet_labels[tid]
                    if label in valid_triplet_labels:
                        triplet_counts[label] += 1

    print(f'\nTotal boxes processed: {n_boxes:,}')
    print(f'Unique valid duplexes seen: {len(duplex_counts)} / 49')
    print(f'Unique valid triplets seen: {len(triplet_counts)} / 86')

    # Add zero-count combos (valid but never seen)
    for label in valid_duplex_labels:
        if label not in duplex_counts:
            duplex_counts[label] = 0
    for label in valid_triplet_labels:
        if label not in triplet_counts:
            triplet_counts[label] = 0

    def corner_case_threshold(counts_dict, abs_thresh, pct):
        vals = sorted(counts_dict.values())
        cutoff_idx = max(0, int(len(vals) * pct / 100) - 1)
        pct_cutoff = vals[cutoff_idx] if vals else 0
        return max(abs_thresh, pct_cutoff)

    d_cutoff = corner_case_threshold(duplex_counts,  args.threshold, args.pct)
    t_cutoff = corner_case_threshold(triplet_counts, args.threshold, args.pct)

    rows = []
    for label, cnt in sorted(duplex_counts.items(), key=lambda x: x[1]):
        rows.append({
            'type': 'duplex',
            'combination': label,
            'count': cnt,
            'is_corner_case': cnt <= d_cutoff,
        })
    for label, cnt in sorted(triplet_counts.items(), key=lambda x: x[1]):
        rows.append({
            'type': 'triplet',
            'combination': label,
            'count': cnt,
            'is_corner_case': cnt <= t_cutoff,
        })

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'combination', 'count', 'is_corner_case'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved → {args.out}')

    # Summary table
    d_rows = [r for r in rows if r['type'] == 'duplex']
    t_rows = [r for r in rows if r['type'] == 'triplet']
    d_corner = [r for r in d_rows if r['is_corner_case']]
    t_corner = [r for r in t_rows if r['is_corner_case']]

    print('\n' + '='*60)
    print('COMBINATION FREQUENCY SUMMARY')
    print('='*60)
    print(f'{"Type":<10} {"Total valid":>12} {"Corner cases":>13} {"Threshold":>10}')
    print('-'*60)
    print(f'{"duplex":<10} {len(d_rows):>12} {len(d_corner):>13} {d_cutoff:>10}')
    print(f'{"triplet":<10} {len(t_rows):>12} {len(t_corner):>13} {t_cutoff:>10}')
    print('='*60)

    print('\nTop 10 rarest DUPLEXES (corner cases):')
    print(f'  {"Combination":<35} {"Count":>8}')
    for r in d_corner[:10]:
        print(f'  {r["combination"]:<35} {r["count"]:>8,}')

    print('\nTop 10 rarest TRIPLETS (corner cases):')
    print(f'  {"Combination":<55} {"Count":>8}')
    for r in t_corner[:10]:
        print(f'  {r["combination"]:<55} {r["count"]:>8,}')


if __name__ == '__main__':
    main()
