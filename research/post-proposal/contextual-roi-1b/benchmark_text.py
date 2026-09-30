"""Verify the frozen official InternVL text tower and encode the same 184 phrases.

Uses the author's tokenizer/config and explicit frozen LoRA branches so every
stored tensor is accounted for. Does not accept missing pretrained parameters.
"""
import argparse, hashlib, json, time, tempfile, shutil, os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from legacy_rotary import LegacyRotaryEmbedding


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
    # Hash while staging once to node-local scratch; avoid repeated25GB NFS
    # scans for verification followed by mmap page faults during model loading.
    scratch_base = os.environ.get('SLURM_TMPDIR', '/tmp')
    scratch = None
    if shutil.disk_usage(scratch_base).free > text_path.stat().st_size + (4 << 30):
        scratch = tempfile.TemporaryDirectory(prefix='road-1b-text-', dir=scratch_base)
        local_path = Path(scratch.name)/text_path.name
        h = hashlib.sha256(); copied = 0; report_at = time.monotonic()
        with text_path.open('rb') as source, local_path.open('wb') as target:
            for chunk in iter(lambda: source.read(8 << 20), b''):
                target.write(chunk); h.update(chunk); copied += len(chunk)
                if time.monotonic()-report_at > 30:
                    print('TEXT_STAGE_PROGRESS', copied, by_kind['text']['bytes'], flush=True)
                    report_at = time.monotonic()
        assert copied == by_kind['text']['bytes'] and h.hexdigest() == by_kind['text']['sha256']
        text_path = local_path
        print('TEXT_CHECKPOINT_STAGED_AND_VERIFIED', copied, flush=True)
    else:
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
    rotary_keys = {f'layers.{i}.self_attn.rotary_emb.inv_freq' for i in range(32)}
    assert {k for k in state if k.endswith('.rotary_emb.inv_freq')} == rotary_keys
    for key in rotary_keys:
        value = state[key]
        assert value.shape == (64,) and torch.isfinite(value).all() and (value > 0).all(), key
        assert torch.equal(value, weights['transformer.' + key]), 'Base/delta rotary mismatch: ' + key
    lora = {}
    for key in list(state):
        if '.lora_A.' in key or '.lora_B.' in key:
            module, suffix = key.split('.lora_')
            side, end = suffix.split('.', 1)
            assert end in ('weight', 'default.weight'), key
            lora.setdefault(module, {})[side] = state.pop(key)
    expected_modules = {f'layers.{i}.self_attn.{q}' for i in range(32) for q in ('q_proj', 'v_proj')}
    assert set(lora) == expected_modules, sorted(lora)
    with torch.device('meta'):
        model = LlamaModel(cfg)
    # Real CPU buffers reproduce the publisher-pinned4.28 initialization/load
    # order and avoid leaving nonpersistent caches on the meta device.
    for layer in model.layers:
        layer.self_attn.rotary_emb = LegacyRotaryEmbedding(128, cfg.max_position_embeddings)
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
              'rotary_semantics': 'transformers4.28.1 init-before-load cache; all32 checkpoint buffers preserved',
              'seconds_encoding': time.monotonic()-start, 'phrase_bank_sha256': sha(dest),
              'source_phrase_order_sha256': hashlib.sha256(json.dumps(original['order']).encode()).hexdigest(),
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}
    (root/'results/text-benchmark.json').write_text(json.dumps(result, indent=2))
    print('TEXT_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__': main()
