"""
Script 2: map_examples.py

Maps Dr. Moradi's 9 advisor scenario examples to the closest valid
duplex/triplet labels in ROAD-Waymo using fuzzy label name matching.

Usage:
    python map_examples.py [--anno PATH] [--counts PATH] [--out PATH]

Defaults:
    --anno      /data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json
    --counts    combination_counts.csv   (output of count_combinations.py)
    --out       example_mapping.csv
"""

import json
import csv
import argparse
from difflib import SequenceMatcher


# Dr. Moradi's 9 advisor examples — hardcoded
ADVISOR_EXAMPLES = [
    {
        'id': 1,
        'text': 'School bus discharging children; must stop even if lane is clear.',
        'keywords': ['bus', 'stop', 'busstop'],
    },
    {
        'id': 2,
        'text': 'Cyclist signaling left turn with arm; reduce speed if coming in front.',
        'keywords': ['cyc', 'incatlft', 'turnlft', 'movtow'],
    },
    {
        'id': 3,
        'text': 'Pedestrian looking at phone, not traffic; unlikely to cross yet.',
        'keywords': ['ped', 'wait2x', 'stop', 'pav'],
    },
    {
        'id': 4,
        'text': 'Pedestrian looking at road; may cross.',
        'keywords': ['ped', 'xing', 'wait2x', 'xingfmlft', 'xingfmrht'],
    },
    {
        'id': 5,
        'text': 'Car brakes suddenly for a logical reason.',
        'keywords': ['car', 'brake', 'stop', 'vehlane'],
    },
    {
        'id': 6,
        'text': 'Waymo stiff rule: stops for plastic bag — VLM provides semantic nuance.',
        'keywords': ['smalveh', 'stop', 'mov', 'vehlane'],
    },
    {
        'id': 7,
        'text': 'Ambulance with sirens behind car; need to pull over.',
        'keywords': ['emveh', 'mov', 'movtow', 'incomlane'],
    },
    {
        'id': 8,
        'text': 'Road blocked; human gesturing left; must deviate from GPS path and turn left.',
        'keywords': ['ped', 'turlft', 'incatlft', 'jun'],
    },
    {
        'id': 9,
        'text': 'Long-tail scenario: school bus not seen enough times; model reasons through logic.',
        'keywords': ['bus', 'stop', 'busstop', 'wait2x'],
    },
]


def fuzzy_score(text, label):
    """Score how well a label matches a set of keywords."""
    label_lower = label.lower().replace('-', '').replace('_', '')
    text_lower  = text.lower().replace('-', '').replace('_', '').replace(' ', '')
    return SequenceMatcher(None, text_lower, label_lower).ratio()


def keyword_score(keywords, label):
    """Score based on keyword presence in label (case-insensitive)."""
    label_lower = label.lower()
    hits = sum(1 for kw in keywords if kw.lower() in label_lower)
    return hits / max(len(keywords), 1)


def classify_match(score):
    if score >= 0.5:
        return 'direct_match'
    elif score >= 0.2:
        return 'partial_match'
    else:
        return 'not_representable'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--anno',   default='/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json')
    p.add_argument('--counts', default='combination_counts.csv')
    p.add_argument('--out',    default='example_mapping.csv')
    return p.parse_args()


def load_counts(path):
    counts = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                counts[row['combination']] = int(row['count'])
    except FileNotFoundError:
        pass
    return counts


def main():
    args = parse_args()

    with open(args.anno) as f:
        data = json.load(f)

    valid_duplex_labels  = data['duplex_labels']   # list of 49
    valid_triplet_labels = data['triplet_labels']  # list of 86
    all_labels = (
        [('duplex',  l) for l in valid_duplex_labels] +
        [('triplet', l) for l in valid_triplet_labels]
    )

    counts = load_counts(args.counts)

    rows = []
    for ex in ADVISOR_EXAMPLES:
        best_label = None
        best_type  = None
        best_score = -1.0

        for ltype, label in all_labels:
            score = keyword_score(ex['keywords'], label) * 0.7 + fuzzy_score(ex['text'], label) * 0.3
            if score > best_score:
                best_score = score
                best_label = label
                best_type  = ltype

        match_type = classify_match(best_score)
        freq = counts.get(best_label, 'N/A')

        rows.append({
            'advisor_example_id':  ex['id'],
            'advisor_example':     ex['text'],
            'closest_label':       best_label,
            'label_type':          best_type,
            'match_type':          match_type,
            'match_score':         round(best_score, 3),
            'frequency_in_dataset': freq,
        })

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'advisor_example_id', 'advisor_example', 'closest_label',
            'label_type', 'match_type', 'match_score', 'frequency_in_dataset',
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved → {args.out}')

    # Summary table
    print('\n' + '='*100)
    print('ADVISOR EXAMPLE → ROAD-WAYMO LABEL MAPPING')
    print('='*100)
    print(f'  {"ID":>3}  {"Match":>16}  {"Score":>6}  {"Freq":>8}  {"Closest Label":<35}  Example')
    print('-'*100)
    for r in rows:
        print(f'  {r["advisor_example_id"]:>3}  {r["match_type"]:>16}  {r["match_score"]:>6.3f}  '
              f'{str(r["frequency_in_dataset"]):>8}  {r["closest_label"]:<35}  {r["advisor_example"][:60]}')
    print('='*100)

    # representability summary
    by_match = {}
    for r in rows:
        by_match[r['match_type']] = by_match.get(r['match_type'], 0) + 1
    print('\nRepresentability summary:')
    for k, v in sorted(by_match.items()):
        print(f'  {k}: {v}')

    print('\nNote: "not_representable" scenarios require VLM reasoning beyond '
          'ROAD-Waymo structured labels — prime motivation for Approach 3.')


if __name__ == '__main__':
    main()
