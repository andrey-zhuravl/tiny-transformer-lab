"""Hyperbolic Transformer language model."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ttlab.utils.geometry import exp0, log0
from ttlab.utils.masking import causal_mask

from .base import BaseTTLabModel
from .heads import LMHead
from .registry import register_model


class HyperBlock(nn.Module):
    """Transformer block executed in the tangent (Euclidean) space."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * ff_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        h = self.ln_1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        h = self.ln_2(x)
        x = x + self.dropout(self.mlp(h))
        return x


@register_model("hyper")
class HyperLM(BaseTTLabModel):
    """Hyperbolic Poincaré-ball Transformer LM."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        max_seq_len: int,
        curvature_c: float,
        proj_tau: float,
        tie_embeddings: bool,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [HyperBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = LMHead(
            d_model,
            vocab_size,
            tie_embeddings=tie_embeddings,
            embed_weight=self.token_embedding.weight if tie_embeddings else None,
        )
        self.max_seq_len = max_seq_len
        self.curvature = float(curvature_c)
        self.proj_tau = float(proj_tau)

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, object],
        *,
        vocab_size: int,
        n_labels: Optional[int] = None,
    ) -> "HyperLM":
        del n_labels  # Not used for LM
        model_cfg = cfg.get("model")
        if not isinstance(model_cfg, Mapping):
            raise TypeError("Model configuration must be a mapping")
        hyper_cfg = model_cfg.get("hyper")
        if not isinstance(hyper_cfg, Mapping):
            raise TypeError("Hyperbolic configuration must be a mapping")
        return cls(
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            ff_mult=int(model_cfg.get("ff_mult", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            curvature_c=float(hyper_cfg.get("curvature_c", 1.0)),
            proj_tau=float(hyper_cfg.get("proj_tau", 1.0)),
            tie_embeddings=bool(model_cfg.get("tie_embeddings", True)),
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input_ids = batch["input_ids"]
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 2D tensor")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        tangent = self.proj_tau * hidden
        hyper = exp0(tangent, self.curvature)

        attn_mask = causal_mask(input_ids.size(1), device=input_ids.device)
        attention_mask = batch.get("attention_mask")
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        tangent = log0(hyper, self.curvature)
        for block in self.blocks:
            tangent = block(
                tangent,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            hyper = exp0(tangent, self.curvature)
            tangent = log0(hyper, self.curvature)

        final_hidden = self.final_norm(tangent)
        logits = self.head(final_hidden)
        return {"logits": logits}

    def compute_loss(self, batch: Dict[str, Tensor], outputs: Dict[str, Tensor]) -> Tensor:
        logits = outputs["logits"]
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        if labels is None:
            labels = batch["input_ids"][:, 1:]
            logits = logits[:, :-1]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 1:]
        else:
            seq_len = labels.size(1)
            logits = logits[:, :seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seq_len]

        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        target = labels.reshape(-1)
        if attention_mask is not None:
            ignore_index = -100
            target = target.clone()
            target[attention_mask.reshape(-1) == 0] = ignore_index
            loss = F.cross_entropy(logits, target, ignore_index=ignore_index)
        else:
            loss = F.cross_entropy(logits, target)
        return loss
