"""Utilities for constructing graph-derived attention biases."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import torch
from torch import Tensor

__all__ = ["bias_from_edges"]


def _normalise_edge(edge: Sequence[int | float]) -> tuple[int, int, float]:
    if len(edge) < 2:
        raise ValueError("Edges must specify at least source and destination indices")
    i = int(edge[0])
    j = int(edge[1])
    weight = float(edge[2]) if len(edge) >= 3 else 1.0
    return i, j, weight


def bias_from_edges(
    seq_len: int,
    edges: Iterable[Sequence[int | float]] | None,
    *,
    bias_type: str = "adjacency",
    scale: float = 0.5,
    clip: float = 5.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Construct an additive attention bias matrix from edge information."""

    dtype = dtype or torch.float32
    bias = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    if not edges:
        return bias

    if bias_type == "adjacency":
        for entry in edges:
            i, j, weight = _normalise_edge(entry)
            if 0 <= i < seq_len and 0 <= j < seq_len:
                bias[i, j] = bias[i, j] + scale * weight
    elif bias_type == "shortest_path":
        neighbours: list[list[int]] = [[] for _ in range(seq_len)]
        for entry in edges:
            i, j, _ = _normalise_edge(entry)
            if 0 <= i < seq_len and 0 <= j < seq_len:
                neighbours[i].append(j)
                neighbours[j].append(i)
        for src in range(seq_len):
            dist = [-1] * seq_len
            dist[src] = 0
            queue: deque[int] = deque([src])
            while queue:
                node = queue.popleft()
                for nxt in neighbours[node]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[node] + 1
                        queue.append(nxt)
            for dst, value in enumerate(dist):
                if dst == src or value <= 0:
                    continue
                bias[src, dst] = min(bias[src, dst], -scale * float(value))
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")

    if clip > 0:
        bias = torch.clamp(bias, -clip, clip)
    return bias
