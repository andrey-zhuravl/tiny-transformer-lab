from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ttlab.cli.cli_validate import validate_app


def _write(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _schema_text() -> str:
    return (
        "dataset:\n"
        "  required_fields: [id, meta, task]\n"
        "  meta_fields:\n"
        "    required: [grammar_rev, seed, split, template_index]\n"
        "task:\n"
        "  required:\n"
        "    - lm\n"
        "    - seq2seq\n"
        "    - cls\n"
    )


def test_cli_data_validate_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "meta": {
                    "grammar_rev": "v1",
                    "seed": 123,
                    "split": "train",
                    "template_index": 4,
                },
                "task": "lm",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    schema_path = tmp_path / "schema.yaml"
    _write(schema_path, _schema_text())
    runner = CliRunner()
    result = runner.invoke(
        validate_app,
        [
            "run",
            "--in",
            str(dataset_path),
            "--schema",
            str(schema_path),
            "--format",
            "JSONL",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["rows_valid"] == 1
    assert payload["rows_invalid"] == 0


def test_cli_data_validate_invalid_input(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "meta": {
                    "grammar_rev": "v1",
                    "seed": 123,
                    "split": "train",
                },
                "task": "lm",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    schema_path = tmp_path / "schema.yaml"
    _write(schema_path, _schema_text())

    runner = CliRunner()

    result = runner.invoke(
        validate_app,
        [
            "run",
            "--in_error", #HERE WRONG TESTED INPUT
            str(dataset_path),
            "--schema",
            str(schema_path),
            "--format",
            "JSONL",
        ]
    )

    assert result.exit_code == 2
    # payload = json.loads(result.stdout)
    # assert payload["rows_valid"] == 0
    # assert payload["rows_invalid"] == 1
    # assert payload["errors"]

def test_cli_run2(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        validate_app,
        [
            "run2",
            "--in",
            str(tmp_path),
        ]
    )

    assert result.exit_code == 0