import json
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("typer")

from typer.testing import CliRunner

from ttlab.cli.cli_tokenizer import app

def _write_dataset(tmp_path: Path) -> Path:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"task":"lm","text":"hello world"}\n', encoding="utf-8")
    dev.write_text('{"task":"lm","text":"world"}\n', encoding="utf-8")
    test.write_text('{"task":"lm","text":"hi"}\n', encoding="utf-8")
    manifest = tmp_path / "dataset.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "toy",
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


def test_tok_train_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    dataset_manifest = _write_dataset(tmp_path)
    out_dir = tmp_path / "artifacts"
    result = runner.invoke(
        app,
        [
            "train",
            "--dataset-manifest",
            str(dataset_manifest),
            "--algo",
            "bpe",
            "--vocab-size",
            "64",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert Path(payload["tokenizer_path"]).exists()

    inspect_result = runner.invoke(
        app,
        [
            "inspect",
            "--tokenizer", payload["tokenizer_path"],
            "--dataset-manifest", str(dataset_manifest),
            "--out", str(out_dir / "inspect"),
        ],
    )
    assert inspect_result.exit_code == 0
    inspect_payload = json.loads(inspect_result.stdout)
    assert "report" in inspect_payload
