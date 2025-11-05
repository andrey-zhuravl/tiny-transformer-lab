"""Hyperbolic geometry helpers for the Poincaré ball model."""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["proj_to_ball", "exp0", "log0", "safe_norm"]

_EPS = 1e-7


def safe_norm(x: Tensor, *, dim: int = -1, keepdim: bool = True) -> Tensor:
    """Return the L2 norm of ``x`` clamped away from zero for numerical stability."""

    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
    return norm.clamp_min(_EPS)


def _ball_radius(c: float) -> float:
    if c <= 0:
        raise ValueError("Curvature must be positive for the Poincaré ball")
    return 1.0 / math.sqrt(c)


def proj_to_ball(x: Tensor, c: float, *, eps: float = 1e-5) -> Tensor:
    """Project ``x`` onto the closed Poincaré ball of curvature ``c``."""

    radius = _ball_radius(c)
    norm = safe_norm(x)
    max_norm = (1.0 - eps) * radius
    scale = torch.minimum(torch.ones_like(norm), max_norm / norm)
    return x * scale


def _safe_div(num: Tensor, denom: Tensor) -> Tensor:
    return num / denom.clamp_min(_EPS)


def exp0(v: Tensor, c: float) -> Tensor:
    """Exponential map at the origin for the Poincaré ball."""

    radius = math.sqrt(c)
    v_norm = safe_norm(v)
    scaled_norm = radius * v_norm
    coef = _safe_div(torch.tanh(scaled_norm), scaled_norm)
    coef = torch.where(v_norm > 0, coef, torch.ones_like(coef))
    x = v * coef
    return proj_to_ball(x, c)


def _artanh(z: Tensor) -> Tensor:
    z = torch.clamp(z, -1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(z) - torch.log1p(-z))


def log0(x: Tensor, c: float) -> Tensor:
    """Logarithmic map at the origin for the Poincaré ball."""

    radius = math.sqrt(c)
    x_norm = safe_norm(x)
    scaled_norm = torch.clamp(radius * x_norm, max=1 - 1e-6)
    coef = _safe_div(_artanh(scaled_norm), scaled_norm)
    coef = torch.where(x_norm > 0, coef, torch.ones_like(coef))
    return x * coef
