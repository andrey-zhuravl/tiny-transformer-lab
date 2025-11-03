from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")

from ttlab.train.runner import evaluate_checkpoint, train


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _generate_rows(n: int, vocab: int, min_len: int = 4, max_len: int = 12):
    rng = random.Random(13)
    rows = []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        rows.append({"input_ids": [rng.randint(1, vocab - 1) for _ in range(length)]})
    return rows


def test_smoke_train_and_eval(tmp_path) -> None:
    data_dir = Path(tmp_path)
    train_rows = _generate_rows(64, vocab=32)
    dev_rows = _generate_rows(16, vocab=32)
    _write_jsonl(data_dir / "train.jsonl", train_rows)
    _write_jsonl(data_dir / "dev.jsonl", dev_rows)

    cfg = {
        "model": {
            "kind": "vanilla",
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "ff_mult": 2,
            "dropout": 0.0,
            "max_seq_len": 64,
            "pos_enc": "sinus",
            "tie_embeddings": True,
            "head": {"kind": "lm"},
        },
        "trainer": {
            "batch_size": 16,
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "max_steps": 40,
            "grad_clip": 1.0,
            "grad_ckpt": False,
            "log_interval": 10,
            "eval_interval": 20,
            "num_workers": 0,
            "device": "cpu",
        },
        "tokenizer": {"vocab_size": 32, "pad_token_id": 0},
    }

    checkpoint = train(data_dir, cfg, seed=212, mlflow_uri=None)
    assert checkpoint.exists()

    metrics = evaluate_checkpoint(checkpoint, data_dir, cfg)
    assert metrics["loss"] > 0
    assert metrics["ppl"] > 0
