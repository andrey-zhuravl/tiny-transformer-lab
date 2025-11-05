"""Unit tests for linear attention utilities."""

from __future__ import annotations

import torch
from torch import nn

from ttlab.models.linear import LinearBlock, LinearLM
from ttlab.utils.feature_maps import phi


def test_phi_non_negative() -> None:
    tensor = torch.randn(4, 5, 6)
    assert torch.all(phi(tensor, "elu_plus_one") >= 0)
    assert torch.all(phi(tensor, "relu") >= 0)


def test_linear_forward_shapes_and_loss() -> None:
    model = LinearLM(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        max_seq_len=128,
        linear_cfg={"phi": "elu_plus_one", "n_features": 0, "causal": True, "eps": 1e-6},
        tie_embeddings=True,
    )
    batch = {
        "input_ids": torch.randint(0, 100, (2, 32)),
        "attention_mask": torch.ones(2, 32, dtype=torch.long),
    }
    outputs = model(batch)
    assert outputs["logits"].shape == (2, 32, 100)
    loss = model.compute_loss(batch, outputs)
    assert torch.isfinite(loss)


def _configured_block(causal: bool) -> LinearBlock:
    block = LinearBlock(
        d_model=4,
        n_heads=1,
        ff_mult=2,
        dropout=0.0,
        n_features=0,
        phi="elu_plus_one",
        causal=causal,
        eps=1e-6,
    )
    block.ln_attn = nn.Identity()  # type: ignore[assignment]
    block.ln_mlp = nn.Identity()  # type: ignore[assignment]
    for module in block.mlp:  # type: ignore[arg-type]
        if isinstance(module, nn.Linear):
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()
    block.out_proj.weight.copy_(torch.eye(4))
    block.q_proj.weight.zero_()
    block.k_proj.weight.zero_()
    block.v_proj.weight.copy_(torch.eye(4))
    return block


def test_linear_noncausal_uniform_attention_matches_average() -> None:
    block = _configured_block(causal=False)
    x = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.long)
    output = block(x, mask)
    expected = x.mean(dim=1, keepdim=True).expand_as(x)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_linear_causal_matches_prefix_average() -> None:
    block = _configured_block(causal=True)
    x = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    mask = torch.ones(1, 4, dtype=torch.long)
    output = block(x, mask)
    expected = torch.stack([x[:, :i + 1, :].mean(dim=1) for i in range(4)], dim=1)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
