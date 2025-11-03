"""Decoder-only Transformer language model implementation."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ttlab.utils.masking import causal_mask

from .base import BaseTTLabModel
from .heads import LMHead
from .registry import register_model


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    """Single Transformer decoder block."""

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


@register_model("vanilla")
class VanillaLM(BaseTTLabModel):
    """Decoder-only Transformer language model."""

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
        pos_enc: str,
        tie_embeddings: bool,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        if rotary:
            raise ValueError("rotary positional encodings are not implemented in A1 milestone")
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if pos_enc == "sinus":
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        elif pos_enc == "learned":
            self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_enc}")
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = LMHead(
            d_model,
            vocab_size,
            tie_embeddings=tie_embeddings,
            embed_weight=self.token_embedding.weight if tie_embeddings else None,
        )
        self.max_seq_len = max_seq_len
        self.pos_enc = pos_enc
        self.vocab_size = vocab_size

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, object],
        *,
        vocab_size: int,
        n_labels: Optional[int] = None,
    ) -> "VanillaLM":
        del n_labels  # Unused for language modelling
        model_cfg = cfg.get("model")
        if not isinstance(model_cfg, Mapping):  # pragma: no cover - defensive
            raise TypeError("Model configuration must be a mapping")
        return cls(
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            ff_mult=int(model_cfg.get("ff_mult", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            pos_enc=str(model_cfg.get("pos_enc", "sinus")),
            tie_embeddings=bool(model_cfg.get("tie_embeddings", True)),
            rotary=bool(model_cfg.get("rotary", False)),
        )

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input_ids = batch["input_ids"]
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 2D tensor of shape [batch, seq_len]")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds configured max_seq_len={self.max_seq_len}"
            )
        x = self.token_embedding(input_ids)
        if self.pos_enc == "sinus":
            x = self.positional_encoding(x)
        else:
            positions = torch.arange(x.size(1), device=x.device)
            x = x + self.positional_encoding(positions)
        x = self.dropout(x)

        attn_mask = causal_mask(x.size(1), device=x.device)
        key_padding_mask = None
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

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
            loss = F.cross_entropy(logits, target, ignore_index=ignore_index)
        else:
            loss = F.cross_entropy(logits, target)
        return loss
