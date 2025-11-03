"""Causal language modelling head supporting tied embeddings."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn


class LMHead(nn.Module):
    """Projection layer producing vocabulary logits from hidden states."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        tie_embeddings: bool = True,
        embed_weight: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            if embed_weight is None:
                raise ValueError("Embedding weight must be provided when tie_embeddings=True")
            if embed_weight.shape != self.proj.weight.shape:
                raise ValueError(
                    "Embedding weight shape does not match projection weight shape for tying"
                )
            self.proj.weight = embed_weight
        else:
            nn.init.normal_(self.proj.weight, mean=0.0, std=d_model**-0.5)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits."""

        return self.proj(hidden_states)
