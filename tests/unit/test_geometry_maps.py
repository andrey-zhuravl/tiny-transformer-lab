"""Unit tests for hyperbolic geometry helpers."""

from __future__ import annotations

import torch

from ttlab.utils.geometry import exp0, log0, proj_to_ball


def test_exp_log_roundtrip() -> None:
    torch.manual_seed(0)
    vectors = torch.randn(8, 5) * 0.1
    curvature = 1.0
    projected = exp0(vectors, curvature)
    recovered = log0(projected, curvature)
    reproj = exp0(recovered, curvature)
    assert torch.allclose(projected, reproj, atol=1e-4, rtol=1e-4)


def test_projection_within_ball() -> None:
    curvature = 0.5
    radius = (1.0 / curvature**0.5) - 1e-6
    tensor = torch.randn(3, 7) * 10
    projected = proj_to_ball(tensor, curvature)
    norms = projected.norm(dim=-1)
    assert torch.all(norms < radius)
