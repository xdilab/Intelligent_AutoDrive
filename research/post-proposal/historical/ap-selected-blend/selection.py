"""Deterministic selection using only internal-development triplet AP."""
def choose(candidates):
    assert candidates and len({x['weight'] for x in candidates})==len(candidates)
    assert all(0<=x['weight']<=1 for x in candidates)
    return max(candidates,key=lambda x:(x['dev_triplet_crop_AP'],-x['weight']))
def validate_selection(s):
    assert s['metric']=='internal-development mean AP over all 86 triplets'
    for seed in s['seeds'].values():
        for expert in ['phrase','shuffled']:
            v=seed[expert];assert [c['weight'] for c in v['candidates']]==s['grid'];assert v['selected_weight']==choose(v['candidates'])['weight']
