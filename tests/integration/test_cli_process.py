from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ttlab.core.validate import ExitCode
from ttlab.utils.paths import get_project_path
from ttlab.cli.cli_process import process_app

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"row-{index}",
            "meta": {
                "grammar_rev": "g.rev",
                "seed": index,
                "split": "train",
                "template_index": index,
            },
            "task": "lm",
        }
        for index in range(10)
    ]


def test_cli_data_process_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, _sample_rows())

    out_dir = tmp_path / "out"
    metrics_dir = tmp_path / "metrics"
    runner = CliRunner()
    result = runner.invoke(
        process_app,
        [
            "run",
            "--in", str(dataset_path),
            "--schema", str(get_project_path("conf/data/sample_dataset.yaml")),
            "--format", "JSONL",
            "--split", "train=0.7,dev=0.2,test=0.1",
            "--seed", "17",
            "--out", str(out_dir),
            "--metrics", str(metrics_dir),
        ]
    )

    assert result.exit_code == 0
    assert (out_dir / "dataset.manifest.json").exists()
    assert (out_dir / "dataset_stats.json").exists()
    for split in ("train", "dev", "test"):
        assert (out_dir / f"{split}.jsonl").exists()

    metrics_file = metrics_dir / "metrics.jsonl"
    assert metrics_file.exists()
    assert metrics_file.read_text().strip()


def test_cli_data_process_invalid_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, _sample_rows())

    bad_schema = tmp_path / "schema.yaml"
    bad_schema.write_text("invalid: true\n")
    runner = CliRunner()

    result = runner.invoke(
        process_app,
        [
            "run",
            "--in", str(dataset_path),
            "--schema", str(bad_schema),
            "--format", "JSONL",
            "--split", "train=0.7,dev=0.2,test=0.1",
            "--seed", "17",
        ]
    )

    assert result.exit_code == ExitCode.INVALID_INPUT.value
