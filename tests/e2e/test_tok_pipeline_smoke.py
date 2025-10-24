import json
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from ttlab.core.tokenizer import inspect_tokenizer, train_or_import_tokenizer
from ttlab.registry import REGISTRY_PATH
from ttlab.utils.files import read_json


def _write_dataset(tmp_path: Path) -> Path:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"task":"lm","text":"b1 smoke"}\n', encoding="utf-8")
    dev.write_text('{"task":"lm","text":"dev"}\n', encoding="utf-8")
    test.write_text('{"task":"lm","text":"test"}\n', encoding="utf-8")
    manifest = tmp_path / "dataset.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "toy-smoke",
                "splits": {
                    "train": {"path": str(train)},
                    "dev": {"path": str(dev)},
                    "test": {"path": str(test)},
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_dataset_manifest_to_tok_train_to_tok_inspect_mlflow_smoke(tmp_path: Path) -> None:
    dataset_manifest = _write_dataset(tmp_path)
    out_dir = tmp_path / "artifacts"
    train_result = train_or_import_tokenizer(
        dataset_manifest=dataset_manifest,
        algo="bpe",
        vocab_size=64,
        norm="nfc",
        lower=True,
        punct_policy="keep",
        seed=5,
        out_dir=out_dir,
    )
    inspect_dir = tmp_path / "inspect"
    inspect_result = inspect_tokenizer(
        tokenizer_path=Path(train_result["tokenizer_path"]),
        dataset_manifest=dataset_manifest,
        out_dir=inspect_dir,
    )
    registry = read_json(REGISTRY_PATH)
    assert registry["tokenizers"], "Registry should contain at least one tokenizer entry"
    assert Path(inspect_result["report_path"]).exists()
    assert Path(train_result["manifest_path"]).exists()
