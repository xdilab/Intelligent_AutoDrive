"""Transformers4.28.1 LLaMA rotary-cache behavior, with a4.37 output shape.

Adapted from Hugging Face Transformers (Apache-2.0):
https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/llama/modeling_llama.py

The official InternVideo2 dependencies pin4.28.1. That implementation creates
nonpersistent FP32 cosine/sine caches BEFORE loading checkpoint inv_freq.
Short inputs use those caches; replacing them with caches recomputed from the
BF16 checkpoint frequencies would change the original inference behavior.
"""
import torch


class LegacyRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        self._build_cache(max_position_embeddings)

    def _build_cache(self, length, device=None):
        t = torch.arange(length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if device is not None:
            emb = emb.to(device)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            self._build_cache(seq_len, x.device)
        #4.37 expects [sequence, head_dim], versus4.28's [1,1,sequence,head_dim].
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)
