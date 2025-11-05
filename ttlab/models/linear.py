"""Linear attention model built with positive feature maps."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ttlab.models.base import BaseTTLabModel
from ttlab.models.heads import LMHead
from ttlab.models.registry import register_model
from ttlab.utils.feature_maps import phi as feature_map


class LinearBlock(nn.Module):
    """Transformer block using linear attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        *,
        n_features: int,
        phi: str,
        causal: bool,
        eps: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for linear attention")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.feature_dim = n_features if n_features > 0 else self.d_head
        self.phi_kind = phi
        self.causal = causal
        self.eps = eps
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        if self.feature_dim != self.d_head:
            self.q_feature = nn.Linear(self.d_head, self.feature_dim, bias=False)
            self.k_feature = nn.Linear(self.d_head, self.feature_dim, bias=False)
        else:
            self.q_feature = nn.Identity()
            self.k_feature = nn.Identity()
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_mlp = nn.LayerNorm(d_model)
        hidden_dim = d_model * ff_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _shape_heads(self, tensor: Tensor) -> Tensor:
        bsz, seq_len, _ = tensor.shape
        return tensor.view(bsz, seq_len, self.n_heads, self.d_head)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        bsz, seq_len, _ = x.shape
        residual = x
        h = self.ln_attn(x)
        q = self._shape_heads(self.q_proj(h))
        k = self._shape_heads(self.k_proj(h))
        v = self._shape_heads(self.v_proj(h))
        q_features = feature_map(self.q_feature(q), kind=self.phi_kind, eps=self.eps)
        k_features = feature_map(self.k_feature(k), kind=self.phi_kind, eps=self.eps)
        if attention_mask is not None:
            mask = attention_mask.to(q_features.dtype).unsqueeze(-1).unsqueeze(-1)
            k_features = k_features * mask
            v = v * mask
        if not self.causal:
            kv = torch.einsum("bthf,bthd->bhfd", k_features, v)
            k_sum = torch.einsum("bthf->bhf", k_features)
            numerators = torch.einsum("bthf,bhfd->bthd", q_features, kv)
            denominators = torch.einsum("bthf,bhf->bth", q_features, k_sum).unsqueeze(-1)
            attn_out = numerators / (denominators + self.eps)
        else:
            k_perm = k_features.permute(0, 2, 1, 3)
            v_perm = v.permute(0, 2, 1, 3)
            kv = torch.einsum("bhtf,bhtd->bhtfd", k_perm, v_perm).cumsum(dim=2)
            k_cumsum = k_perm.cumsum(dim=2)
            q_perm = q_features.permute(0, 2, 1, 3)
            numerators = torch.einsum("bhtf,bhtfd->bhtd", q_perm, kv)
            denominators = torch.einsum("bhtf,bhtf->bht", q_perm, k_cumsum).unsqueeze(-1)
            attn_out = (numerators / (denominators + self.eps)).permute(0, 2, 1, 3)
        attn_out = attn_out.reshape(bsz, seq_len, self.d_model)
        x = residual + self.dropout(self.out_proj(attn_out))
        residual = x
        h = self.ln_mlp(x)
        x = residual + self.dropout(self.mlp(h))
        return x


@register_model("linear")
class LinearLM(BaseTTLabModel):
    """Language model built with linear attention blocks."""

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
        linear_cfg: Mapping[str, object],
        tie_embeddings: bool,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                LinearBlock(
                    d_model,
                    n_heads,
                    ff_mult,
                    dropout,
                    n_features=int(linear_cfg.get("n_features", 0)),
                    phi=str(linear_cfg.get("phi", "elu_plus_one")),
                    causal=bool(linear_cfg.get("causal", True)),
                    eps=float(linear_cfg.get("eps", 1e-6)),
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
    ) -> "LinearLM":
        del n_labels
        model_cfg = cfg.get("model")
        if not isinstance(model_cfg, Mapping):  # pragma: no cover - defensive
            raise TypeError("Model configuration must be a mapping")
        linear_cfg = model_cfg.get("linear", {})
        if not isinstance(linear_cfg, Mapping):  # pragma: no cover - defensive
            raise TypeError("Linear model configuration must be a mapping")
        return cls(
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            ff_mult=int(model_cfg.get("ff_mult", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
            linear_cfg=linear_cfg,
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
