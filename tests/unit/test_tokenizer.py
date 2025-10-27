import json
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from ttlab.core.tokenizer import SPECIAL_TOKENS, train_or_import_tokenizer
from ttlab.utils.files import read_json


def _write_dataset(tmp_path: Path) -> Path:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"task":"lm","text":"Hello world"}\n{"task":"lm","text":"Tokenizer"}\n', encoding="utf-8")
    dev.write_text('{"task":"lm","text":"dev sample"}\n', encoding="utf-8")
    test.write_text('{"task":"lm","text":"test sample"}\n', encoding="utf-8")
    manifest = tmp_path / "dataset.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "toy-v1",
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


def test_tokenizer_train_char_bpe_unigram_small(tmp_path: Path) -> None:
    dataset_manifest = _write_dataset(tmp_path)
    for algo in ("char", "bpe", "unigram"):
        out_dir = tmp_path / algo
        result = train_or_import_tokenizer(
            dataset_manifest=dataset_manifest,
            algo=algo,
            vocab_size=64,
            norm="nfc",
            lower=True,
            punct_policy="keep",
            seed=42,
            out_dir=out_dir,
        )
        manifest = result["manifest"]
        assert manifest["algo"] == algo
        assert Path(result["tokenizer_path"]).exists()
        assert Path(result["manifest_path"]).exists()
        assert Path(result["report_path"]).exists()


def test_manifest_hashes(tmp_path: Path) -> None:
    dataset_manifest = _write_dataset(tmp_path)
    out_dir = tmp_path / "bpe"
    result = train_or_import_tokenizer(
        dataset_manifest=dataset_manifest,
        algo="bpe",
        vocab_size=64,
        norm="nfc",
        lower=False,
        punct_policy="keep",
        seed=1,
        out_dir=out_dir,
    )
    manifest = result["manifest"]
    assert "tokenizer_json_sha256" in manifest
    assert manifest["input_paths_sha256"]
    loaded_manifest = read_json(result["manifest_path"])
    assert loaded_manifest["tokenizer_json_sha256"] == manifest["tokenizer_json_sha256"]


def test_special_tokens_present(tmp_path: Path) -> None:
    from tokenizers import Tokenizer

    dataset_manifest = _write_dataset(tmp_path)
    out_dir = tmp_path / "unigram"
    result = train_or_import_tokenizer(
        dataset_manifest=dataset_manifest,
        algo="unigram",
        vocab_size=64,
        norm="none",
        lower=False,
        punct_policy="keep",
        seed=7,
        out_dir=out_dir,
    )
    tokenizer = Tokenizer.from_file(result["tokenizer_path"])
    for token in SPECIAL_TOKENS:
        assert tokenizer.token_to_id(token) is not None
