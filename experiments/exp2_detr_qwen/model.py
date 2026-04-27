from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from position_encoding import SpatiotemporalPositionEncoding


class QwenViTExtractor(nn.Module):
    """
    Extracts per-frame spatial feature maps from Qwen2.5-VL's frozen ViT.

    Why Qwen's ViT specifically:
        Qwen2.5-VL-7B was pretrained on massive vision-language data. Its ViT
        already understands "pedestrian", "vehicle", "traffic light" without any
        fine-tuning — we get rich semantic features for free. We only LoRA-adapt
        the first 8 ViT blocks to the ROAD-Waymo visual domain.

    Frozen vs trainable:
        The ViT backbone is frozen by default; only the LoRA delta weights in the
        first 8 blocks are trainable. This keeps GPU memory manageable (no full
        7B backward pass) while still allowing domain adaptation.

    Output:
        List of T tensors, each [H', W', D] where:
          T  = number of frames in the clip (8 for exp2)
          H' = height after spatial merge (input_height / (patch_size * merge_size))
          W' = width  after spatial merge
          D  = 3584 (Qwen2.5-VL-7B ViT hidden dim)

    Copied from exp1b so we can load the same warm-start checkpoint.
    """

    def __init__(self, model_id: str, freeze: bool = True):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration

        # Load the full 7B model just to extract its visual encoder.
        # device_map="cpu" avoids placing it on GPU during init — caller moves it.
        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        # model.visual contains: patch_embed → transformer blocks → merger (projector).
        # We only use the visual encoder output; the LLM layers are discarded.
        self.visual = full.model.visual
        del full  # free the rest of the 7B model immediately

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)

    def add_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list | None = None,
        n_layers: int = 8,
    ) -> None:
        """
        Wrap the first n_layers ViT attention blocks with LoRA adapters.

        Why only the first 8 layers:
            Early ViT layers encode low-level features (edges, textures) that
            transfer well across domains. Deep layers encode high-level semantics
            already aligned to language — LoRA-ing them risks disturbing the rich
            pre-trained representations. 8/32 is a common heuristic.

        LoRA mechanics (Hu et al. 2022):
            Each targeted Linear(in, out) is replaced by:
                W_output = W_frozen @ x + (B @ A) @ x * (alpha/r)
            where A ∈ R^(r×in), B ∈ R^(out×r), r << in.
            Only A and B are trained; W_frozen is never updated.
            This reduces trainable params from in*out to r*(in+out).
            For a 3584-dim layer: 3584*3584=12.8M → r*(3584+3584)=114K at r=8.

        Target modules — ViT attention uses combined qkv + proj:
            "qkv" = one Linear that computes Q,K,V together (3×dim output)
            "proj" = output projection after attention
        """
        # Workaround for an upstream torch dtype bug that can appear on some setups
        if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn"):
            torch.float8_e8m0fnu = torch.float8_e4m3fn

        from peft import LoraConfig, get_peft_model

        # Build a regex that matches only the first n_layers blocks.
        # PEFT's target_modules can accept a regex string.
        block_range = "|".join(str(i) for i in range(n_layers))
        attn_modules = "|".join(target_modules or ["qkv", "proj"])
        regex = rf"blocks\.({block_range})\.attn\.({attn_modules})"

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=regex,
            bias="none",       # don't add LoRA to bias terms
            inference_mode=False,
        )
        self.visual = get_peft_model(self.visual, lora_config)

    def lora_parameters(self):
        """Returns only the trainable LoRA delta weights for the optimizer."""
        return [p for p in self.visual.parameters() if p.requires_grad]

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            pixel_values:    [N_patches, 1176] — Qwen's packed patch format.
                             Each patch is 14×14×6 = 1176 values (×6 because the
                             spatial merge step combines 2×2 patches before the ViT).
            image_grid_thw:  [N_images, 3] — (T, H_patches, W_patches) per image,
                             telling the model how to un-pack patches back to frames.

        Returns: List of T tensors, each [H', W', D].

        How Qwen's spatial merge works:
            After the ViT transformer blocks, Qwen applies a spatial merge with
            merge_size=2: adjacent 2×2 tokens are concatenated and projected,
            reducing spatial resolution by 2× in each axis.
            So if the ViT patch grid is (H_patches, W_patches), the merged grid
            is (H_patches//2, W_patches//2) = (H', W').
        """
        out = self.visual(pixel_values, image_grid_thw)
        # pooler_output is [total_merged_tokens, D] — all frames concatenated
        merged = out.pooler_output

        # Recover merge_size from the model (avoids hard-coding 2)
        base = getattr(self.visual, "base_model", self.visual)
        base = getattr(base, "model", base)
        merge_size = base.spatial_merge_size  # = 2 for Qwen2.5-VL

        frame_feats: List[torch.Tensor] = []
        offset = 0
        for row in image_grid_thw:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            # After spatial merge: merged grid is (h//merge_size, w//merge_size)
            h_m, w_m = h // merge_size, w // merge_size
            # Total tokens for this image: T frames × H' × W'
            n = t * h_m * w_m
            # Slice this image's tokens and reshape to [T, H', W', D]
            chunk = merged[offset : offset + n].view(t, h_m, w_m, -1)
            for frame in chunk:
                frame_feats.append(frame)  # each is [H', W', D]
            offset += n

        return frame_feats


class FeatureProjection(nn.Module):
    """
    Projects ViT features from D=3584 down to d_model=256 for the DETR decoder.

    Why project down:
        3584-dim cross-attention would be extremely memory-hungry (attention matrix
        scales with d^2). 256 is the standard DETR decoder dimension — large enough
        to carry semantic content, small enough for efficient attention over 2048 tokens.

    LayerNorm after projection:
        Stabilises the scale of features going into the decoder. Without it,
        the random initialisation of the Linear can produce activations that are
        orders of magnitude off from what the decoder expects.

    Cast to float32:
        The ViT runs in bfloat16. The DETR decoder attention is numerically
        sensitive (softmax over 2048 values — large denominators lose precision
        in bfloat16). Projecting to float32 here prevents attention collapse.
    """

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x.float()))  # cast bfloat16 → float32


class MLP(nn.Module):
    """
    Simple feed-forward network used as the per-frame box prediction head.

    num_layers=3 means: Linear → ReLU → Linear → ReLU → Linear (no ReLU on output).
    The final Linear outputs 4 values (cx, cy, w, h) before sigmoid is applied
    externally in DETRDecoder to map to [0,1] normalised coordinates.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, num_layers: int = 3):
        super().__init__()
        dims = [d_in] + [d_hidden] * (num_layers - 1) + [d_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderLayer(nn.Module):
    """
    One layer of the DETR Transformer decoder. Stacked 6× in DETRDecoder.

    Architecture (pre-norm variant):
        queries → LayerNorm → self-attention  → residual
                → LayerNorm → cross-attention → residual
                → LayerNorm → FFN            → residual

    Pre-norm vs post-norm:
        Original Transformer uses post-norm (normalise after residual). Pre-norm
        (normalise before the sub-layer, i.e. on the residual branch) gives more
        stable gradients, especially for deep stacks. DETR implementations
        commonly use pre-norm.

    Self-attention (queries attending to queries):
        Allows queries to co-ordinate with each other — if query 3 is already
        covering a pedestrian on the left, query 7 can learn to attend to a
        different part of the scene. This prevents duplicate detections.

    Cross-attention (queries attending to memory):
        Each query "reads" the 2048 spatiotemporal ViT tokens. This is where the
        detection actually happens — the query's attention weights highlight the
        spatial regions and frames relevant to the agent it's tracking.

    Position is injected into memory but NOT into queries:
        memory + pos_embed is passed as the key/value in cross-attention.
        The positional signal is thus available during the spatial lookup
        without biasing the query representations themselves.
    """

    def __init__(self, d_model: int, nhead: int, dim_ffn: int, dropout: float):
        super().__init__()
        # Three separate LayerNorms — one per sub-layer, per pre-norm convention
        self.norm_q1 = nn.LayerNorm(d_model)  # before self-attention
        self.norm_q2 = nn.LayerNorm(d_model)  # before cross-attention
        self.norm_q3 = nn.LayerNorm(d_model)  # before FFN

        # batch_first=True: input shape [batch, seq, d_model] instead of [seq, batch, d_model]
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),  # 256 → 1024 (4× expansion, standard)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),  # 1024 → 256
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries:   [1, N_queries, d_model] — current query representations
            memory:    [1, T*H*W, d_model]    — projected ViT features (all frames)
            pos_embed: [1, T*H*W, d_model]    — spatiotemporal position encoding

        Returns: updated queries [1, N_queries, d_model]
        """
        # --- Self-attention: queries read from each other ---
        q = self.norm_q1(queries)
        sa_out, _ = self.self_attn(q, q, q, need_weights=False)
        queries = queries + self.dropout(sa_out)  # residual connection

        # --- Cross-attention: queries read from memory tokens ---
        q = self.norm_q2(queries)
        # Position is added to memory keys/values so the query's attention
        # pattern is spatially informed. The query is not position-augmented
        # (it hasn't yet "decided" where in the scene it lives).
        mem = memory + pos_embed
        ca_out, _ = self.cross_attn(q, mem, mem, need_weights=False)
        queries = queries + self.dropout(ca_out)

        # --- FFN: non-linear mixing within each query independently ---
        q = self.norm_q3(queries)
        queries = queries + self.dropout(self.ffn(q))
        return queries


class DETRDecoder(nn.Module):
    """
    Clip-level DETR decoder: 100 learnable queries attend to 2048 spatiotemporal
    tokens and produce one tube prediction per query.

    Why clip-level (one pass for all T frames):
        Temporal context is available for free — each query's cross-attention
        pattern can span all 8 frames simultaneously. A per-frame detector sees
        one frame at a time; our decoder sees the entire clip. This is analogous
        to the temporal receptive field of 3D convolutions in 3D-RetinaNet, but
        based on attention rather than learned kernels.

    Why 100 queries:
        ROAD-Waymo clips have at most ~30 annotated agent tubes. 100 gives 3×
        headroom so the matching always has enough free queries. The excess
        (unmatched) queries learn to output near-zero agentness confidence.

    Per-frame box prediction via temporal_embed:
        After 6 decoder layers, each query has a single [d_model] representation
        that summarises "which agent am I tracking, and what does it look like?"
        But we need T separate boxes (one per frame). We condition the shared MLP
        on a learned temporal embedding for each frame:
            box[t] = sigmoid(box_head(query_feat + temporal_embed[t]))
        The temporal embed shifts the query feature so the MLP produces a
        different box for each frame — i.e., the query "tracks" its agent across
        the clip.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ffn: int,
        dropout: float,
        num_queries: int,
        clip_len: int,
    ):
        super().__init__()
        # Learnable query embeddings — these are the "slots" that become detections.
        # Initialised randomly; the decoder learns to specialise each slot.
        self.query_embed = nn.Embedding(num_queries, d_model)
        # One learned vector per frame index — used to condition per-frame box prediction
        self.temporal_embed = nn.Embedding(clip_len, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_ffn, dropout) for _ in range(num_layers)]
        )
        # Shared box MLP: (query_feat + temporal_embed[t]) → 4 values, then sigmoid
        self.box_head = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, memory: torch.Tensor, pos_embed: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            memory:    [T*H*W, d_model] — projected ViT tokens (2048 for 8-frame clip)
            pos_embed: [T*H*W, d_model] — spatiotemporal position encoding
            t:         number of frames (T=8 for exp2)

        Returns:
            query_feats: [N_queries, d_model] — final query representations (for cls heads)
            pred_boxes:  [N_queries, T, 4]    — sigmoid box coords per query per frame
        """
        # query_embed.weight is [N_queries, d_model]; add batch dim for MHA (batch_first)
        queries = self.query_embed.weight.unsqueeze(0)   # [1, N_queries, d_model]
        memory = memory.unsqueeze(0)                      # [1, T*H*W, d_model]
        pos_embed = pos_embed.unsqueeze(0)                # [1, T*H*W, d_model]

        # 6 decoder layers: each refines queries by attending to memory
        for layer in self.layers:
            queries = layer(queries, memory, pos_embed)

        query_feats = queries.squeeze(0)  # [N_queries, d_model]

        # Per-frame box prediction: same MLP, different temporal conditioning per frame
        boxes = []
        for frame_idx in range(t):
            temp = self.temporal_embed.weight[frame_idx].unsqueeze(0)  # [1, d_model]
            # query_feats + temp broadcasts across queries → [N_queries, d_model]
            boxes.append(torch.sigmoid(self.box_head(query_feats + temp)))
        pred_boxes = torch.stack(boxes, dim=1)  # [N_queries, T, 4] in [cx,cy,w,h] ∈ [0,1]
        return query_feats, pred_boxes


class ClassificationHeads(nn.Module):
    """
    Five independent linear classifiers applied to each query's final representation.

    Why five heads (agent / action / loc / duplex / triplet):
        ROAD++ uses a compositional label hierarchy. Each level describes the same
        agent at increasing specificity:
            agent    (10 classes)  — what type: Ped, Car, LarVeh, ...
            action   (22 classes)  — what it's doing: Xing, Wait2X, Moving, ...
            loc      (16 classes)  — where: RhtPav, OppLane, ...
            duplex   (49 classes)  — agent+action combination
            triplet  (86 classes)  — agent+action+location combination
        Having all five lets us compute constraint violations (some combinations
        are logically impossible — t-norm loss penalises those).

    Independent sigmoid vs softmax:
        Multi-label problem (a pedestrian can be Xing AND on RhtPav). We use
        sigmoid per class so each class is an independent binary prediction.
        Softmax would force exactly one class per head, which is wrong here.

    Float32 cast:
        Classification head logits go into focal loss which uses log() — needs
        float32 precision for numerical stability.
    """

    def __init__(self, d_model: int, head_sizes: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_model, size) for name, size in head_sizes.items()}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Returns raw logits — sigmoid applied externally (during loss or inference)
        return {name: head(x.float()) for name, head in self.heads.items()}


class DETRROADModel(nn.Module):
    """
    Full Exp2 model: Qwen ViT → projection → DETR decoder → classification heads.

    Data flow for one 8-frame clip:
        1. ViT extracts per-frame features:    8 × [H', W', 3584]
        2. Flatten to token sequence:          [2048, 3584]
        3. Project to decoder dim:             [2048, 256]
        4. Add spatiotemporal position:        [2048, 256]
        5. DETR decoder (6 layers):            queries [100, 256] attend to memory [2048, 256]
        6. Per-frame box prediction:           [100, 8, 4]  (100 tubes × 8 frames × 4 coords)
        7. Classification (5 heads):           [100, C] per head
        8. Agentness head:                     [100, 1]  (is this query a real detection?)

    Three optimizer param groups (different LRs):
        - lora_parameters():    ViT LoRA adapters         — LR 5e-5 (pre-trained, small delta)
        - decoder_parameters(): projection + decoder      — LR 1e-4 (fresh init, needs faster LR)
        - head_parameters():    cls heads + agentness     — LR 1e-4 (fresh init)
    """

    def __init__(
        self,
        model_id: str,
        vit_dim: int,
        d_model: int,
        head_sizes: Dict[str, int],
        clip_len: int,
        num_queries: int,
        num_decoder_layers: int,
        nhead: int,
        dim_ffn: int,
        dropout: float,
        freeze_vit: bool = True,
    ):
        super().__init__()
        self.clip_len = clip_len
        self.vit = QwenViTExtractor(model_id, freeze=freeze_vit)
        self.projection = FeatureProjection(vit_dim, d_model)
        self.position = SpatiotemporalPositionEncoding(d_model, max_t=clip_len)
        self.decoder = DETRDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            num_queries=num_queries,
            clip_len=clip_len,
        )
        self.cls_heads = ClassificationHeads(d_model, head_sizes)
        # Separate agentness head: "is this query a real agent or background?"
        # Not included in cls_heads because it has different loss treatment
        # (applied to ALL 100 queries, not just matched ones)
        self.agentness_head = nn.Linear(d_model, 1)

    def add_lora(self, **kwargs) -> None:
        self.vit.add_lora(**kwargs)

    def lora_parameters(self):
        return self.vit.lora_parameters()

    def decoder_parameters(self):
        # projection + position encoding + all decoder layers
        return (
            list(self.projection.parameters())
            + list(self.position.parameters())
            + list(self.decoder.parameters())
        )

    def head_parameters(self):
        return list(self.cls_heads.parameters()) + list(self.agentness_head.parameters())

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Step 1-2: ViT feature extraction → list of [H', W', 3584] per frame
        frame_feats = self.vit(pixel_values, image_grid_thw)
        if not frame_feats:
            raise RuntimeError("QwenViTExtractor returned no frame features")

        t = len(frame_feats)
        h, w, _ = frame_feats[0].shape  # spatial grid size (H', W')

        # Flatten all frames to one token sequence: [T*H'*W', 3584]
        flat_tokens = torch.cat([f.reshape(-1, f.shape[-1]) for f in frame_feats], dim=0)

        # Step 3: project 3584 → 256, cast to float32
        memory = self.projection(flat_tokens)   # [T*H'*W', 256]

        # Step 4: add spatiotemporal position encoding
        pos = self.position(t=t, h=h, w=w, device=memory.device).to(memory.dtype)

        # Step 5-6: DETR decoder → query features and per-frame boxes
        query_feats, pred_boxes = self.decoder(memory.float(), pos.float(), t=t)
        # query_feats: [100, 256]   pred_boxes: [100, 8, 4]

        # Step 7: classification heads (raw logits, no sigmoid yet)
        pred_logits = self.cls_heads(query_feats)           # {head: [100, C]}

        # Step 8: agentness (also raw logits)
        pred_logits["agentness"] = self.agentness_head(query_feats.float())  # [100, 1]

        return {
            "pred_boxes": pred_boxes,      # [100, T, 4]  in [cx,cy,w,h] ∈ [0,1]
            "pred_logits": pred_logits,    # {head: [100, C]}  raw logits
            "query_feats": query_feats,    # [100, 256]  decoder output (debug/aux use)
            "T": t,                        # number of frames
            "frame_shape": (h, w),         # spatial grid dims (H', W')
        }
