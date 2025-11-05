"""Smoke tests for the Hyperbolic language model."""

from __future__ import annotations

import torch

from ttlab.models.hyper import HyperLM


def test_hyper_forward_and_loss() -> None:
    model = HyperLM(
        vocab_size=101,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        max_seq_len=64,
        curvature_c=1.0,
        proj_tau=1.0,
        tie_embeddings=True,
    )
    batch = {"input_ids": torch.randint(0, 100, (2, 10))}
    outputs = model(batch)
    assert outputs["logits"].shape == (2, 10, 101)
    loss = model.compute_loss(batch, outputs)
    assert torch.isfinite(loss)
