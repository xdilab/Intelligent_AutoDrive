"""Embed the 184 class phrases with the frozen CLIP_S text tower.

Order matches the 184 layout exactly: [agentness | agent 10 | action 22 |
loc 16 | duplex 49 | triplet 86], phrases from phrases.json (label order from
the dataset JSON). Output: phrase_embeds.pt {"embeds": [184, D] fp32
L2-normalized, "order": [class names]}.
"""
import json, sys
from pathlib import Path
import torch
sys.stdout.reconfigure(line_buffering=True)
torch.cuda.set_per_process_memory_fraction(0.10, 0)

E12 = Path(__file__).resolve().parent
REPO = "OpenGVLab/InternVideo2_CLIP_S"

def load_clip_s(device):  # verbatim from cache_clip_feats.py (module runs argparse on import)
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    cls = get_class_from_dynamic_module(
        "modeling_internvideo2encoder.InternVideo2_CLIP_small", REPO)
    m = cls(cfg)
    missing, unexpected = m.load_state_dict(
        load_file(hf_hub_download(REPO, "model.safetensors")), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m.half().to(device).eval()

d = json.load(open("/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"))
import os
ph = json.load(open(os.environ.get("PHRASES", str(E12 / "phrases.json"))))
order, texts = ["agentness"], [ph["agentness"]]
for h, labels in (("agent", d["agent_labels"]), ("action", d["action_labels"]),
                  ("loc", d["loc_labels"]), ("duplex", d["duplex_labels"]),
                  ("triplet", d["triplet_labels"])):
    for n in labels:
        order.append(f"{h}:{n}"); texts.append(ph[h][n])
assert len(texts) == 184

device = torch.device("cuda:0")
model = load_clip_s(device)
tok = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
tok = {k: v.to(device) for k, v in tok.items()} if isinstance(tok, dict) else tok.to(device)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    emb = model.encode_text(tok).float()
emb = torch.nn.functional.normalize(emb, dim=-1).cpu()
torch.save({"embeds": emb, "order": order}, os.environ.get("EMBEDS_OUT", str(E12 / "phrase_embeds.pt")))
print(f"[embed] {emb.shape} saved; sample cos(Ped, Ped-MovTow) = "
      f"{float(emb[order.index('agent:Ped')] @ emb[order.index('duplex:Ped-MovTow')]):.3f}, "
      f"cos(Ped, Car) = {float(emb[order.index('agent:Ped')] @ emb[order.index('agent:Car')]):.3f}")
