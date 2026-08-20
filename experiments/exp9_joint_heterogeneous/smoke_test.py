"""Exp9 smoke test — verify every component before any real run.

Checks (each prints PASS/detail):
  1. constraints_verified.json loads; penalty sets sized 171 duplex / 3434 triplet.
  2. CorrectedTNorm: uniform p=0.5 logits give a positive loss at lam=1;
     a one-hot valid composition (Ped+MovAway+Jun hot, everything else -9)
     yields a (near-)zero loss; an INVALID one (Ped+Red) yields a larger loss.
  3. RoadDataset[0]: image opens, boxes normalized in [0,1], targets [n,184],
     at least one all-zeros (negative) row across the first few frames.
  4. LangDataset[0]: paths exist, prompt/target non-empty.
  5. Exp9Model: LoRA lands on both ViT and LLM (asserted in ctor);
     road_forward returns [n,184] finite logits;
     ROAD loss backward puts nonzero grads on head AND ViT LoRA, none on LLM LoRA;
     lang_forward loss finite; backward puts nonzero grads on LLM LoRA.
  6. One full train.py cycle via --max-cycles (run manually, see bottom).
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

import torch

import config as C
from dataset import RoadDataset, LangDataset
from losses import CorrectedTNorm, RoadCriterion, AGENT_OFF, ACTION_OFF, LOC_OFF


def check_tnorm():
    tn = CorrectedTNorm(lam=1.0)
    assert tn.inv_d.shape[0] == 220 - 49, tn.inv_d.shape
    assert tn.inv_t.shape[0] == 3520 - 86, tn.inv_t.shape

    uniform = torch.zeros(4, C.NUM_CLASSES)          # sigmoid → 0.5 everywhere
    l_uniform = float(tn(uniform))
    assert l_uniform > 0.5, l_uniform                # min(0.5,0.5)*2 terms = 1.0

    valid = torch.full((1, C.NUM_CLASSES), -9.0)
    valid[0, AGENT_OFF + 0] = 9.0                    # Ped
    valid[0, ACTION_OFF + 3] = 9.0                   # MovAway
    valid[0, LOC_OFF + 10] = 9.0                     # Jun  (Ped-MovAway-Jun is valid)
    l_valid = float(tn(valid))

    invalid = torch.full((1, C.NUM_CLASSES), -9.0)
    invalid[0, AGENT_OFF + 0] = 9.0                  # Ped
    invalid[0, ACTION_OFF + 0] = 9.0                 # Red — Ped-Red is invalid
    l_invalid = float(tn(invalid))
    assert l_invalid > l_valid * 10, (l_valid, l_invalid)
    print(f"PASS tnorm: uniform={l_uniform:.3f} valid={l_valid:.5f} "
          f"invalid={l_invalid:.5f}")


def check_road_data():
    ds = RoadDataset()
    img, boxes, tgts = ds[0]
    assert boxes.min() >= 0.0 and boxes.max() <= 1.0
    assert tgts.shape == (boxes.shape[0], C.NUM_CLASSES)
    neg = sum(int((ds[i][2].sum(1) == 0).any()) for i in range(5))
    assert neg > 0, "no all-zeros negative rows in first 5 frames — suspicious"
    print(f"PASS road data: frame0 {boxes.shape[0]} boxes, img {img.size}, "
          f"negatives present in {neg}/5 frames")
    return ds


def check_lang_data():
    bddx = LangDataset(C.BDDX_TRAIN_JSON, "bddx")
    covla = LangDataset(C.COVLA_TRAIN_JSON, "covla")
    for ds in (bddx, covla):
        path, user, tgt = ds[0]
        from pathlib import Path
        assert Path(path).exists(), path
        assert user and tgt and "<image>" not in user
    print("PASS lang data")
    return bddx


def check_model(road_ds, lang_ds):
    device = torch.device("cuda:0")
    from model import Exp9Model
    model = Exp9Model(device)
    crit = RoadCriterion().to(device)
    crit.tnorm.lam = 1.0

    img, boxes, tgts = road_ds[0]
    logits = model.road_forward(img, boxes)
    assert logits.shape == (boxes.shape[0], C.NUM_CLASSES)
    assert torch.isfinite(logits).all()
    loss, parts = crit(logits, tgts.to(device))
    loss.backward()

    g_head = float(model.head.weight.grad.abs().sum())
    g_vit = sum(float(p.grad.abs().sum()) for n, p in model.vlm.named_parameters()
                if p.grad is not None and "lora" in n and "visual" in n)
    g_llm = sum(float(p.grad.abs().sum()) for n, p in model.vlm.named_parameters()
                if p.grad is not None and "lora" in n and "visual" not in n)
    assert g_head > 0 and g_vit > 0, (g_head, g_vit)
    assert g_llm == 0, f"ROAD leg leaked gradient into LLM LoRA: {g_llm}"
    print(f"PASS road fwd/bwd: loss={float(loss):.4f} "
          f"(focal={float(parts['focal']):.4f} tnorm={float(parts['tnorm']):.5f}) "
          f"grads head={g_head:.3g} vit={g_vit:.3g} llm={g_llm}")
    model.vlm.zero_grad(set_to_none=True)
    model.head.zero_grad(set_to_none=True)

    lloss = model.lang_forward(*lang_ds[0])
    assert torch.isfinite(lloss)
    lloss.backward()
    g_llm = sum(float(p.grad.abs().sum()) for n, p in model.vlm.named_parameters()
                if p.grad is not None and "lora" in n and "visual" not in n)
    assert g_llm > 0
    print(f"PASS lang fwd/bwd: loss={float(lloss):.4f} llm-lora grad={g_llm:.3g}")


if __name__ == "__main__":
    check_tnorm()
    road_ds = check_road_data()
    lang_ds = check_lang_data()
    check_model(road_ds, lang_ds)
    print("\nAll smoke checks passed. Now run 2 full cycles:")
    print("  CUDA_VISIBLE_DEVICES=0 python -u train.py --max-cycles 2 --tag smoke")
