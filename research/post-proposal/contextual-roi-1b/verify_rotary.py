"""Compare the compatibility adapter to the unmodified upstream4.28.1 class.

Pass --reference modeling_llama.py downloaded from the URL in legacy_rotary.py
and --checkpoint the pinned1B_clip.pth. Reference source is SHA-256 pinned.
"""
import argparse, ast, hashlib, json
from pathlib import Path
import torch
from legacy_rotary import LegacyRotaryEmbedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    a = ap.parse_args()
    raw = a.reference.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '61affbfa550d004bb1256e88c8e69d6d1e17e7ef6ac30c553e478d093ec089fe'
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlamaRotaryEmbedding')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(a.reference), 'exec'), ns)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == 'e60e0a6e05daec48285254b17b0b91c7d814b3ac74e538b2c62201553eeec6b1'
    ck = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    rotary = [(k, v) for k, v in ck.items() if k.endswith('rotary_emb.inv_freq')]
    assert len(rotary) == 32
    cases = 0
    for key, value in rotary:
        old = ns['LlamaRotaryEmbedding'](128); new = LegacyRotaryEmbedding(128)
        old.load_state_dict({'inv_freq': value}, strict=True)
        new.load_state_dict({'inv_freq': value}, strict=True, assign=True)
        old.half(); new.half()
        assert torch.equal(old.inv_freq, new.inv_freq)
        for length in [1, 80, 2048, 2050]:
            x = torch.zeros(1, 1, length, 128, dtype=torch.float16)
            oc, os = old(x, seq_len=length); nc, nsin = new(x, seq_len=length)
            assert torch.equal(oc[0, 0], nc) and torch.equal(os[0, 0], nsin), (key, length)
            cases += 1
    print(json.dumps({'passed': True, 'exact_parity_cases': cases, 'checkpoint_buffers_preserved': True}))


if __name__ == '__main__': main()
