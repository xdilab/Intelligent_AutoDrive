"""Verify the frozen official InternVL text tower and encode the same 184 phrases.

Uses the author's tokenizer/config and explicit frozen LoRA branches so every
stored tensor is accounted for. Does not accept missing pretrained parameters.
"""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


class FrozenLoRA(nn.Module):
    def __init__(self, base, a, b):
        super().__init__()
        self.base = base
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        assert a.shape == (16, base.in_features)
        assert b.shape == (base.out_features, 16)

    def forward(self, x):
        # Official r=16, alpha=32; dropout=0.1 is disabled during evaluation.
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args(); root = args.root
    torch.set_num_threads(8)
    assets = json.loads((root/'results/assets.json').read_text())
    assert assets['passed']
    by_kind = {r['kind']: r for r in assets['files']}
    text_path = Path(by_kind['text']['path'])
    assert sha(text_path) == by_kind['text']['sha256']
    tokenizer_path = root/'assets/tokenizer'
    for entry in json.loads((tokenizer_path/'provenance.json').read_text()):
        assert sha(tokenizer_path/entry['file']) == entry['sha256']
    cfg = LlamaConfig.from_pretrained(tokenizer_path, local_files_only=True)
    cfg._attn_implementation = 'sdpa'
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size) == (4096, 32, 49954)
    weights = torch.load(text_path, map_location='cpu', weights_only=False, mmap=True)
    weights = weights.get('module', weights)
    state = {k[len('transformer.'):]: v for k, v in weights.items() if k.startswith('transformer.')}
    projection = weights['text_projection']
    assert projection.shape == (4096, 768)
    # The CLIP delta contains only fixed rotary frequencies on the text side.
    delta = torch.load(by_kind['clip']['path'], map_location='cpu', weights_only=True)
    delta = delta.get('module', delta.get('model', delta))
    for key, value in delta.items():
        if key.startswith('text_encoder.'):
            assert key.endswith('.rotary_emb.inv_freq'), key
            state[key[len('text_encoder.transformer.'):]] = value
    lora = {}
    for key in list(state):
        if '.lora_A.' in key or '.lora_B.' in key:
            module, suffix = key.split('.lora_')
            side, end = suffix.split('.', 1)
            assert end in ('weight', 'default.weight'), key
            lora.setdefault(module, {})[side] = state.pop(key)
        elif key.endswith('.rotary_emb.inv_freq'):
            value = state.pop(key).float()
            expected = 1 / (10000 ** (torch.arange(0, 128, 2).float() / 128))
            assert torch.allclose(value, expected, atol=1e-7, rtol=1e-5), key
    expected_modules = {f'layers.{i}.self_attn.{q}' for i in range(32) for q in ('q_proj', 'v_proj')}
    assert set(lora) == expected_modules, sorted(lora)
    with torch.device('meta'):
        model = LlamaModel(cfg)
    # Copy-on-map tensors retain official checkpoint precision without allocating
    # another randomly initialized 7B model. Strict loading covers every weight.
    model.load_state_dict(state, strict=True, assign=True)
    for path, branches in lora.items():
        assert set(branches) == {'A', 'B'}
        parent, name = path.rsplit('.', 1)
        owner = model.get_submodule(parent)
        setattr(owner, name, FrozenLoRA(getattr(owner, name), branches['A'], branches['B']))
    model.requires_grad_(False).eval().to(device='cuda', dtype=torch.float16)
    projection = projection.cuda().half()
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, local_files_only=True, legacy=False)
    tokenizer.pad_token = ' '; tokenizer.add_eos_token = True
    original = torch.load('/work/bbyrd1/contextual-roi-20260917/data/phrase_embeds.pt', map_location='cpu', weights_only=False)
    phrase_map = json.loads((root/'code/phrases.json').read_text())
    texts = []
    for label in original['order']:
        if label == 'agentness': texts.append(phrase_map['agentness'])
        else:
            head, name = label.split(':', 1); texts.append(phrase_map[head][name])
    assert len(texts) == 184
    ids = tokenizer(['summarize:' + text for text in texts], return_tensors='pt', max_length=80,
                    truncation=True, padding='max_length').input_ids.cuda()
    assert ids.shape == (184, 80) and int(ids.max()) < cfg.vocab_size
    print('TEXT_MODEL_READY', flush=True)
    values = []; start = time.monotonic()
    with torch.inference_mode():
        for i in range(0, 184, 8):
            tokens = ids[i:i+8]; mask = tokens > 0
            hidden = model(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
            pooled = hidden[torch.arange(len(tokens), device='cuda'), mask.sum(1)-1]
            values.append(F.normalize((pooled @ projection).float(), dim=-1).cpu())
    bank = torch.cat(values)
    assert bank.shape == (184, 768) and torch.isfinite(bank).all()
    assert torch.allclose(bank.norm(dim=-1), torch.ones(184), atol=1e-5)
    # Catch collapsed / constant encodings before they enter head training.
    assert bank.std(dim=0).mean() > 1e-5
    output = root/'data'; output.mkdir(exist_ok=True)
    dest = output/'phrase_embeds.pt'; tmp = dest.with_suffix('.tmp')
    torch.save({'embeds': bank, 'order': original['order'], 'texts': texts,
                'text_sha256': by_kind['text']['sha256'], 'tokenizer': json.loads((tokenizer_path/'provenance.json').read_text())}, tmp)
    tmp.replace(dest)
    result = {'passed': True, 'time': time.time(), 'phrases': 184, 'width': 768,
              'weights_strict': True, 'frozen_lora_modules': len(lora),
              'seconds_encoding': time.monotonic()-start, 'phrase_bank_sha256': sha(dest),
              'source_phrase_order_sha256': hashlib.sha256(json.dumps(original['order']).encode()).hexdigest(),
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}
    (root/'results/text-benchmark.json').write_text(json.dumps(result, indent=2))
    print('TEXT_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__': main()
