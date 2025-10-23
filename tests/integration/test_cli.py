from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _write(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _schema_text() -> str:
    return (
        "dataset:\n"
        "  required_fields: [id, meta, task]\n"
        "  meta_fields:\n"
        "    required: [grammar_rev, seed, split, template_index]\n"
        "tasks:\n"
        "  lm:\n"
        "    required: [text]\n"
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
                "task": {"type": "lm", "text": "hello"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    schema_path = tmp_path / "schema.yaml"
    _write(schema_path, _schema_text())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ttlab.cli",
            "data:validate",
            "--in",
            str(dataset_path),
            "--schema",
            str(schema_path),
            "--format",
            "jsonl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
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
                "task": {"type": "lm"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    schema_path = tmp_path / "schema.yaml"
    _write(schema_path, _schema_text())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ttlab.cli",
            "data:validate",
            "--in",
            str(dataset_path),
            "--schema",
            str(schema_path),
            "--format",
            "jsonl",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["rows_valid"] == 0
    assert payload["rows_invalid"] == 1
    assert payload["errors"]
