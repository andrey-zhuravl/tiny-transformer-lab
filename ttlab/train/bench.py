"""Benchmark harness for long-context attention variants."""

from __future__ import annotations

import time
from typing import Dict, Sequence

import psutil
import torch

from ttlab.models import create_model

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover - executed when mlflow missing
    mlflow = None  # type: ignore


def _resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":  # pragma: no cover - GPU specific
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def _build_cfg(
    kind: str,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    seq_len: int,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "model": {
            "kind": kind,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ff_mult": 4,
            "dropout": 0.0,
            "max_seq_len": seq_len,
            "tie_embeddings": True,
        },
        "trainer": {
            "batch_size": 2,
            "device": "auto",
        },
        "tokenizer": {"vocab_size": 32000, "pad_token_id": 0},
    }
    if kind == "linear":
        cfg["model"]["linear"] = {
            "phi": "elu_plus_one",
            "n_features": 0,
            "causal": True,
            "eps": 1e-6,
        }
    elif kind == "sparse":
        cfg["model"]["sparse"] = {
            "pattern": "sliding_window",
            "window": min(128, seq_len),
            "n_global": 4,
            "causal": True,
        }
    return cfg


def _start_mlflow_run(mlflow_uri: str | None, run_name: str):
    if mlflow_uri is None:
        return None
    if mlflow is None:  # pragma: no cover - executed when mlflow missing
        raise RuntimeError("MLflow is not installed. Install tiny-transformer-lab[mlflow].")
    mlflow.set_tracking_uri(mlflow_uri)
    return mlflow.start_run(run_name=run_name)


def _log_nested_params(kind: str, d_model: int, n_layers: int, n_heads: int) -> None:
    if mlflow is None:
        return
    mlflow.log_param("model.kind", kind)
    mlflow.log_param("model.d_model", d_model)
    mlflow.log_param("model.n_layers", n_layers)
    mlflow.log_param("model.n_heads", n_heads)


@torch.no_grad()
def run(
    models: Sequence[str],
    seq_lens: Sequence[int],
    *,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    trials: int = 3,
    device: str | None = None,
    mlflow_uri: str | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run forward benchmarks for the requested models and sequence lengths."""

    device_obj = _resolve_device(device)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    parent_run = _start_mlflow_run(mlflow_uri, "bench")
    if parent_run is not None:
        assert mlflow is not None  # placate type checkers
        mlflow.set_tag("task", "lm")
        mlflow.set_tag("longctx", "true")
        mlflow.set_tag("bench", "true")
        mlflow.log_param("bench.models", ",".join(models))
        mlflow.log_param("bench.seq_lens", ",".join(str(x) for x in seq_lens))
        mlflow.log_param("bench.trials", trials)
        mlflow.log_param("bench.device", str(device_obj))
    try:
        for name in models:
            results[name] = {}
            nested_run = None
            if parent_run is not None:
                nested_run = mlflow.start_run(run_name=f"bench-{name}", nested=True)
                mlflow.set_tag("family", name)
                mlflow.set_tag("task", "lm")
                mlflow.set_tag("longctx", "true")
                _log_nested_params(name, d_model, n_layers, n_heads)
            try:
                for seq_len in seq_lens:
                    cfg = _build_cfg(name, d_model=d_model, n_layers=n_layers, n_heads=n_heads, seq_len=seq_len)
                    model = create_model(cfg, vocab_size=cfg["tokenizer"]["vocab_size"])
                    model.to(device_obj)
                    model.eval()
                    batch = {
                        "input_ids": torch.randint(1, cfg["tokenizer"]["vocab_size"], (2, seq_len), device=device_obj),
                        "attention_mask": torch.ones((2, seq_len), device=device_obj, dtype=torch.long),
                    }
                    model(batch)  # warm-up
                    if device_obj.type == "cuda":  # pragma: no cover - GPU specific
                        torch.cuda.reset_peak_memory_stats(device_obj)
                    start = time.perf_counter()
                    for _ in range(trials):
                        model(batch)
                        if device_obj.type == "cuda":  # pragma: no cover - GPU specific
                            torch.cuda.synchronize(device_obj)
                    elapsed = (time.perf_counter() - start) / max(1, trials)
                    tokens = float(batch["attention_mask"].sum().item())
                    tokens_per_s = tokens / elapsed if elapsed > 0 else float("inf")
                    peak_mb = _peak_memory_mb(device_obj)
                    metrics = {"tokens_per_s": float(tokens_per_s), "peak_mb": float(peak_mb)}
                    results[name][str(seq_len)] = metrics
                    if nested_run is not None:
                        mlflow.log_metric("speed/tokens_per_s", metrics["tokens_per_s"], step=seq_len)
                        mlflow.log_metric("memory/peak_mb", metrics["peak_mb"], step=seq_len)
                    del model, batch
                if device_obj.type == "cuda":  # pragma: no cover - GPU specific
                    torch.cuda.empty_cache()
            finally:
                if nested_run is not None:
                    mlflow.end_run()
        if parent_run is not None:
            mlflow.log_dict(results, artifact_file="bench/results.json")
        return results
    finally:
        if parent_run is not None:
            mlflow.end_run()
