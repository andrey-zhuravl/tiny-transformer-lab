"""Unit tests for graph-derived attention biases."""

from __future__ import annotations

import torch

from ttlab.utils.graph import bias_from_edges


def test_adjacency_bias_weights() -> None:
    bias = bias_from_edges(
        5,
        [[0, 1], [1, 2, 0.5], [3, 4]],
        bias_type="adjacency",
        scale=0.7,
        clip=5.0,
    )
    assert torch.isclose(bias[0, 1], torch.tensor(0.7))
    assert torch.isclose(bias[1, 2], torch.tensor(0.35))
    assert bias[2, 1] == 0  # directional edges


def test_shortest_path_bias_monotonic() -> None:
    bias = bias_from_edges(
        4,
        [[0, 1], [1, 2]],
        bias_type="shortest_path",
        scale=1.0,
        clip=5.0,
    )
    assert torch.isclose(bias[0, 1], torch.tensor(-1.0))
    assert torch.isclose(bias[0, 2], torch.tensor(-2.0))
    assert bias.diagonal().eq(0).all()


def test_bias_clipping() -> None:
    bias = bias_from_edges(
        3,
        [[0, 1], [0, 2]],
        bias_type="adjacency",
        scale=10.0,
        clip=1.5,
    )
    assert torch.all(bias.abs() <= 1.5 + 1e-6)
