from __future__ import annotations

import json
from pathlib import Path

from ttlab.cli import run
from ttlab.core.validate import ExitCode


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
            "task": {"type": "lm", "text": f"hello {index}"},
        }
        for index in range(10)
    ]


def test_cli_data_process_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, _sample_rows())

    out_dir = tmp_path / "out"
    metrics_dir = tmp_path / "metrics"

    exit_code = run(
        [
            "data:process",
            "--in",
            str(dataset_path),
            "--schema",
            str(Path("conf/data/sample_dataset.yaml")),
            "--format",
            "JSONL",
            "--splits",
            "train=0.7",
            "dev=0.2",
            "test=0.1",
            "--seed",
            "17",
            "--out-dir",
            str(out_dir),
            "--metrics-dir",
            str(metrics_dir),
        ]
    )

    assert exit_code is ExitCode.OK
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

    exit_code = run(
        [
            "data:process",
            "--in",
            str(dataset_path),
            "--schema",
            str(bad_schema),
            "--format",
            "JSONL",
            "--splits",
            "train=0.8",
            "test=0.2",
        ]
    )

    assert exit_code is ExitCode.INVALID_INPUT
