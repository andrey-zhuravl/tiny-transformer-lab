"""Utility helpers for constructing attention masks."""

from __future__ import annotations

import torch
from torch import Tensor


def causal_mask(seq_len: int, *, device: torch.device | None = None) -> Tensor:
    """Return an additive causal mask suitable for ``nn.MultiheadAttention``."""

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)
