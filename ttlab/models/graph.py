"""Transformer with graph-structured attention biases."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ttlab.utils.graph import bias_from_edges

from .base import BaseTTLabModel
from .heads import LMHead
from .registry import register_model


class GraphBlock(nn.Module):
    """Decoder block with additive graph-derived attention bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        *,
        bias_type: str,
        bias_scale: float,
        bias_clip: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.bias_type = bias_type
        self.bias_scale = bias_scale
        self.bias_clip = bias_clip

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * ff_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def _reshape(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Optional[Tensor],
        edges: Optional[Sequence[Iterable[Sequence[int | float]]]],
    ) -> Tensor:
        bsz, seq_len, _ = x.shape
        residual = x
        x = self.ln_1(x)
        q = self._reshape(self.q_proj(x))
        k = self._reshape(self.k_proj(x))
        v = self._reshape(self.v_proj(x))

        bias_tensors = []
        if edges is not None and len(edges) == bsz:
            for edge_list in edges:
                bias_tensors.append(
                    bias_from_edges(
                        seq_len,
                        edge_list,
                        bias_type=self.bias_type,
                        scale=self.bias_scale,
                        clip=self.bias_clip,
                        device=x.device,
                        dtype=x.dtype,
                    )
                )
        else:
            bias_tensors = [torch.zeros((seq_len, seq_len), device=x.device, dtype=x.dtype)] * bsz
        bias = torch.stack(bias_tensors, dim=0)

        if attention_mask is not None:
            key_mask = attention_mask == 0
            pad_bias = torch.zeros_like(bias)
            pad_bias.masked_fill_(key_mask.unsqueeze(1), float("-inf"))
            bias = bias + pad_bias

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias.unsqueeze(1),
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        x = residual + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


@register_model("graph")
class GraphLM(BaseTTLabModel):
    """Transformer language model with graph-biased attention."""

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
        bias_type: str,
        bias_scale: float,
        bias_clip: float,
        tie_embeddings: bool,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                GraphBlock(
                    d_model,
                    n_heads,
                    ff_mult,
                    dropout,
                    bias_type=bias_type,
                    bias_scale=bias_scale,
                    bias_clip=bias_clip,
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
    ) -> "GraphLM":
        del n_labels
        model_cfg = cfg.get("model")
        if not isinstance(model_cfg, Mapping):
            raise TypeError("Model configuration must be a mapping")
        graph_cfg = model_cfg.get("graph")
        if not isinstance(graph_cfg, Mapping):
            raise TypeError("Graph configuration must be a mapping")
        bias_cfg = graph_cfg.get("bias")
        if not isinstance(bias_cfg, Mapping):
            raise TypeError("Graph bias configuration must be a mapping")
        return cls(
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            ff_mult=int(model_cfg.get("ff_mult", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            bias_type=str(bias_cfg.get("type", "adjacency")),
            bias_scale=float(bias_cfg.get("scale", 0.5)),
            bias_clip=float(bias_cfg.get("clip", 5.0)),
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

        attention_mask = batch.get("attention_mask")
        edges = batch.get("graph_edges")

        for block in self.blocks:
            hidden = block(
                hidden,
                attention_mask=attention_mask,
                edges=edges,
            )

        hidden = self.final_norm(hidden)
        logits = self.head(hidden)
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
