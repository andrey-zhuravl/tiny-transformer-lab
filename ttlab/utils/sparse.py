"""Sparse attention masking utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _allocate_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.full((seq_len, seq_len), float("-inf"), device=device)


def sliding_window_mask(
    seq_len: int,
    window: int,
    causal: bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return an attention mask permitting tokens within ``window`` distance."""

    if window < 0:
        raise ValueError("window must be non-negative for sliding window attention")
    indices = torch.arange(seq_len, device=device)
    query_index = indices[:, None]
    key_index = indices[None, :]
    distance = key_index - query_index
    allowed = distance.abs() <= window
    if causal:
        allowed &= distance <= 0
    mask = _allocate_mask(seq_len, device)
    mask = mask.masked_fill(allowed, 0.0)
    return mask


def local_global_mask(
    seq_len: int,
    window: int,
    n_global: int,
    causal: bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a mask combining sliding-window attention with global tokens."""

    mask = sliding_window_mask(seq_len, window, causal, device)
    if n_global <= 0:
        return mask
    g = min(n_global, seq_len)
    mask[:, :g] = 0.0
    mask[:g, :] = 0.0
    return mask


class SparseMaskCache:
    """Helper caching masks per (pattern, seq_len, device)."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int, torch.device], torch.Tensor] = {}

    def get(
        self,
        pattern: str,
        seq_len: int,
        device: torch.device,
        *,
        window: int,
        n_global: int,
        causal: bool,
    ) -> torch.Tensor:
        key = (pattern, seq_len, device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if pattern == "sliding_window":
            mask = sliding_window_mask(seq_len, window, causal, device)
        elif pattern == "local_global":
            mask = local_global_mask(seq_len, window, n_global, causal, device)
        else:  # pragma: no cover - validated at call site
            raise ValueError(f"Unknown sparse attention pattern: {pattern}")
        self._cache[key] = mask
        return mask


MASK_CACHE = SparseMaskCache()
