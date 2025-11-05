"""Smoke test for the long-context benchmark utility."""

from __future__ import annotations

from pathlib import Path

from ttlab.train.bench import run


def test_bench_smoke_cpu(tmp_path: Path) -> None:
    results = run(
        ["vanilla", "linear", "sparse"],
        [512],
        d_model=32,
        n_layers=1,
        n_heads=4,
        trials=1,
        device="cpu",
        mlflow_uri=None,
    )
    assert set(results) == {"vanilla", "linear", "sparse"}
    entry = results["vanilla"]["512"]
    assert set(entry) == {"tokens_per_s", "peak_mb"}
    assert entry["tokens_per_s"] > 0
    assert entry["peak_mb"] > 0
