"""Sparse attention model with configurable patterns."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ttlab.models.base import BaseTTLabModel
from ttlab.models.heads import LMHead
from ttlab.models.registry import register_model
from ttlab.utils.sparse import MASK_CACHE


class SparseBlock(nn.Module):
    """Transformer block that uses a structured sparse attention mask."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        *,
        pattern: str,
        window: int,
        n_global: int,
        causal: bool,
    ) -> None:
        super().__init__()
        self.pattern = pattern
        self.window = window
        self.n_global = n_global
        self.causal = causal
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_mlp = nn.LayerNorm(d_model)
        hidden_dim = d_model * ff_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _mask(self, seq_len: int, device: torch.device) -> Tensor:
        return MASK_CACHE.get(
            self.pattern,
            seq_len,
            device,
            window=self.window,
            n_global=self.n_global,
            causal=self.causal,
        )

    def forward(self, x: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        residual = x
        h = self.ln_attn(x)
        attn_mask = self._mask(h.size(1), h.device)
        key_padding = None
        if attention_mask is not None:
            key_padding = attention_mask == 0
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        x = residual + self.dropout(attn_out)
        residual = x
        h = self.ln_mlp(x)
        x = residual + self.dropout(self.mlp(h))
        return x


@register_model("sparse")
class SparseLM(BaseTTLabModel):
    """Language model composed of sparse attention blocks."""

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
        sparse_cfg: Mapping[str, object],
        tie_embeddings: bool,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                SparseBlock(
                    d_model,
                    n_heads,
                    ff_mult,
                    dropout,
                    pattern=str(sparse_cfg.get("pattern", "sliding_window")),
                    window=int(sparse_cfg.get("window", 128)),
                    n_global=int(sparse_cfg.get("n_global", 0)),
                    causal=bool(sparse_cfg.get("causal", True)),
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = LMHead(
            d_model,
            vocab_size,
            tie_embeddings=tie_embeddings,
            embed_weight=self.token_embedding.weight if tie_embeddings else None,
        )
        self.max_seq_len = max_seq_len

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, object],
        *,
        vocab_size: int,
        n_labels: Optional[int] = None,
    ) -> "SparseLM":
        del n_labels
        model_cfg = cfg.get("model")
        if not isinstance(model_cfg, Mapping):  # pragma: no cover - defensive
            raise TypeError("Model configuration must be a mapping")
        sparse_cfg = model_cfg.get("sparse", {})
        if not isinstance(sparse_cfg, Mapping):  # pragma: no cover - defensive
            raise TypeError("Sparse model configuration must be a mapping")
        return cls(
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            ff_mult=int(model_cfg.get("ff_mult", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
            sparse_cfg=sparse_cfg,
            tie_embeddings=bool(model_cfg.get("tie_embeddings", True)),
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input_ids = batch["input_ids"]
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be rank 2: [batch, seq_len]")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        attention_mask = batch.get("attention_mask")
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = self.head(x)
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
            return F.cross_entropy(logits, target, ignore_index=ignore_index)
        return F.cross_entropy(logits, target)
