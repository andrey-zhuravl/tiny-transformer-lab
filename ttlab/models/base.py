"""Base interfaces for Tiny Transformer Lab models."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

import torch
from torch import Tensor, nn


class BaseTTLabModel(nn.Module):
    """Abstract base class implemented by all TTLab models."""

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, object],
        *,
        vocab_size: int,
        n_labels: Optional[int] = None,
    ) -> "BaseTTLabModel":
        """Instantiate the model from a resolved configuration mapping."""

        raise NotImplementedError

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # pragma: no cover - abstract
        """Run the forward pass for a batch of tensors."""

        raise NotImplementedError

    def compute_loss(self, batch: Dict[str, Tensor], outputs: Dict[str, Tensor]) -> Tensor:
        """Compute the loss tensor for the provided batch and model outputs."""

        raise NotImplementedError

    def metrics_spec(self) -> Dict[str, Callable[[Dict[str, Tensor], Dict[str, Tensor]], Tensor]]:
        """Return a mapping of metric names to callables applied during evaluation."""

        return {}

    def param_count(self) -> int:
        """Return the total number of parameters of the model."""

        return int(sum(p.numel() for p in self.parameters()))
