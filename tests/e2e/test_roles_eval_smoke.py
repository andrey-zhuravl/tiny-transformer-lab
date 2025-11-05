"""End-to-end smoke test for role-wise evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ttlab.eval.roles import eval_roles_ckpt
from ttlab.train.runner import train


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_rows(n_samples: int, vocab: int) -> list[dict[str, object]]:
    random.seed(0)
    rows: list[dict[str, object]] = []
    for _ in range(n_samples):
        length = random.randint(6, 9)
        tokens = [random.randint(1, vocab - 1) for _ in range(length)]
        edges = [[idx, idx + 1] for idx in range(length - 1)]
        roles = {"who": [0], "what": [max(1, length // 2)], "where": [length - 1]}
        rows.append({"input_ids": tokens, "graph_edges": edges, "roles": roles})
    return rows


def test_roles_eval_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path
    train_rows = _make_rows(32, vocab=64)
    dev_rows = _make_rows(8, vocab=64)
    _write_jsonl(data_dir / "train.jsonl", train_rows)
    _write_jsonl(data_dir / "dev.jsonl", dev_rows)

    cfg: dict[str, object] = {
        "model": {
            "kind": "graph",
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "ff_mult": 2,
            "dropout": 0.0,
            "max_seq_len": 64,
            "tie_embeddings": True,
            "graph": {
                "bias": {"type": "adjacency", "scale": 0.5, "clip": 5.0},
                "source": {"format": "sample", "path": None},
            },
            "head": {"kind": "lm"},
        },
        "trainer": {
            "batch_size": 4,
            "lr": 3e-4,
            "betas": (0.9, 0.95),
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_steps": 20,
            "grad_clip": 0.0,
            "grad_ckpt": False,
            "log_interval": 10,
            "eval_interval": 100,
            "num_workers": 0,
            "device": "cpu",
        },
        "tokenizer": {"vocab_size": 64, "pad_token_id": 0},
    }

    ckpt_path = train(data_dir, cfg, seed=212, mlflow_uri=None)
    metrics = eval_roles_ckpt(ckpt_path, data_dir)
    assert set(metrics.keys()) == {"who", "what", "where"}
    for value in metrics.values():
        assert value is None or 0.0 <= value <= 1.0
