from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ttlab.models.vanilla import VanillaLM


def test_forward_shapes_and_loss() -> None:
    model = VanillaLM(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        max_seq_len=32,
        pos_enc="sinus",
        tie_embeddings=True,
    )
    batch = {
        "input_ids": torch.randint(0, 100, (2, 10), dtype=torch.long),
        "attention_mask": torch.ones(2, 10, dtype=torch.long),
    }
    outputs = model(batch)
    assert outputs["logits"].shape == (2, 10, 100)
    loss = model.compute_loss(batch, outputs)
    assert torch.isfinite(loss)
