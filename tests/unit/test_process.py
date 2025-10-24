from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttlab.core.process import compute_statistics, perform_splits, process_dataset
from ttlab.core.validate import DATA_FORMAT_JSONL, _read_schema


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _sample_records() -> list[dict[str, object]]:
    return [
        {
            "id": f"row-{index}",
            "meta": {
                "grammar_rev": "g.rev",
                "seed": index,
                "split": "train",
                "template_index": index,
                "is_noisy": index % 2 == 0,
            },
            "task": {
                "type": "cls" if index % 2 else "lm",
                "text": f"text {index}",
                "label": "POS" if index % 2 else "NEG",
            },
        }
        for index in range(6)
    ]


def test_split_determinism(tmp_path: Path) -> None:
    records = _sample_records()
    splits = {"train": 0.5, "dev": 0.3, "test": 0.2}

    first = perform_splits(records, splits=splits, seed=123, output_dir=tmp_path / "first")
    second = perform_splits(records, splits=splits, seed=123, output_dir=tmp_path / "second")

    first_rows = {artifact.name: Path(artifact.path).read_text().splitlines() for artifact in first}
    second_rows = {artifact.name: Path(artifact.path).read_text().splitlines() for artifact in second}

    assert first_rows == second_rows


def test_compute_statistics_correctness() -> None:
    records = _sample_records()
    schema_path = Path("conf/data/sample_dataset.yaml")
    schema = _read_schema(schema_path)

    stats = compute_statistics(records, schema)

    assert stats.rows_total == len(records)
    assert stats.unique_ids == len(records)
    assert stats.task_distribution["lm"] == 3
    assert stats.task_distribution["cls"] == 3
    assert stats.field_coverage["id"] == 1.0
    assert stats.meta_field_coverage["grammar_rev"] == 1.0
    assert stats.label_distribution["cls"]["POS"] == 3
    assert stats.noise_rate == pytest.approx(0.5)


def test_manifest_creation_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, _sample_records())

    schema_path = Path("conf/data/sample_dataset.yaml")
    schema = _read_schema(schema_path)

    result = process_dataset(
        dataset_path=dataset_path,
        schema_path=schema_path,
        schema=schema,
        data_format=DATA_FORMAT_JSONL,
        split_ratios={"train": 0.6, "dev": 0.2, "test": 0.2},
        seed=99,
        output_dir=tmp_path / "out",
        log_to_mlflow=False,
    )

    assert result.manifest_path.exists()
    payload = json.loads(result.manifest_path.read_text())
    assert payload["dataset_id"] == dataset_path.stem
    assert payload["format"] == DATA_FORMAT_JSONL
    assert payload["schema_path"] == str(schema_path)
    assert set(payload["splits"].keys()) == {"train", "dev", "test"}
    for artifact in result.split_artifacts:
        assert artifact.path.exists()
        assert artifact.sha256
