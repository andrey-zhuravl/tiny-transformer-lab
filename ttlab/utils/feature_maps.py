"""Positive feature map utilities for linear attention kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_DEFAULT_EPS = 1e-6


def phi(x: torch.Tensor, kind: str = "elu_plus_one", eps: float = _DEFAULT_EPS) -> torch.Tensor:
    """Apply a positive feature map ``phi`` to the input tensor."""

    if kind == "elu_plus_one":
        return F.elu(x, alpha=1.0) + 1.0
    if kind == "relu":
        return F.relu(x) + eps
    raise ValueError(f"Unknown feature map kind: {kind}")
