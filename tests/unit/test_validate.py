from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from ttlab.core.validate import (
    DATA_FORMAT_JSONL,
    ValidationReport,
    _read_schema,
    validate_dataset,
)


@pytest.fixture()
def sample_schema(tmp_path: Path) -> Path:
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(dedent("""
        dataset:
          required_fields: [id, meta, task]
          meta_fields:
            required: [grammar_rev, seed, split, template_index]
        tasks:
          lm:
            required: [text]
        """).strip())
    return schema_path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_read_schema_ok(sample_schema: Path) -> None:
    schema = _read_schema(sample_schema)
    assert "dataset" in schema
    assert "tasks" in schema


def test_validate_single_record_ok(tmp_path: Path, sample_schema: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "row-1",
                "meta": {
                    "grammar_rev": "v1",
                    "seed": 1,
                    "split": "train",
                    "template_index": 0,
                },
                "task": {"type": "lm", "text": "hello world"},
            }
        ],
    )

    schema = _read_schema(sample_schema)
    report = validate_dataset(dataset, schema, DATA_FORMAT_JSONL)
    assert isinstance(report, ValidationReport)
    assert report.rows_total == 1
    assert report.rows_valid == 1
    assert report.rows_invalid == 0
    assert not report.errors


def test_validate_single_record_missing_field(tmp_path: Path, sample_schema: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "row-1",
                "meta": {
                    "grammar_rev": "v1",
                    "seed": 1,
                    "split": "train",
                },
                "task": {"type": "lm"},
            }
        ],
    )

    schema = _read_schema(sample_schema)
    report = validate_dataset(dataset, schema, DATA_FORMAT_JSONL)
    assert report.rows_total == 1
    assert report.rows_valid == 0
    assert report.rows_invalid == 1
    assert len(report.errors) == 2
    messages = {error.message for error in report.errors}
    assert "Missing meta field 'template_index'" in messages
    assert "Missing task field 'text' for task type 'lm'" in messages
