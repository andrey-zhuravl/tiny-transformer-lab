"""Tests for sparse attention masking utilities."""

from __future__ import annotations

import torch

from ttlab.utils.sparse import MASK_CACHE, local_global_mask, sliding_window_mask


def test_sliding_window_mask_respects_window_and_causality() -> None:
    mask = sliding_window_mask(seq_len=8, window=2, causal=True)
    # Token 5 can attend to positions 3,4,5
    allowed = (mask[5] == 0).nonzero(as_tuple=False).flatten().tolist()
    assert allowed == [3, 4, 5]
    # Future tokens should be masked
    assert mask[5, 6].item() == float("-inf")
    # Non-causal allows symmetric access
    mask_full = sliding_window_mask(seq_len=6, window=1, causal=False)
    assert mask_full[0, 1].item() == 0.0
    assert mask_full[1, 0].item() == 0.0


def test_local_global_mask_exposes_globals() -> None:
    mask = local_global_mask(seq_len=6, window=1, n_global=2, causal=True)
    # Everyone can attend to globals
    assert torch.all(mask[:, :2] == 0.0)
    # Globals can attend everywhere
    assert torch.all(mask[:2] == 0.0)
    # Non-global tokens still respect causality outside the window
    assert mask[4, 5].item() == float("-inf")


def test_sparse_mask_cache_returns_same_tensor_for_same_device() -> None:
    cache_mask_first = MASK_CACHE.get(
        "sliding_window",
        4,
        torch.device("cpu"),
        window=1,
        n_global=0,
        causal=True,
    )
    cache_mask_second = MASK_CACHE.get(
        "sliding_window",
        4,
        torch.device("cpu"),
        window=1,
        n_global=0,
        causal=True,
    )
    assert cache_mask_first is cache_mask_second
