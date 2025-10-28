from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ttlab.cli.cli_process import process_app
from ttlab.core.validate import ExitCode
from ttlab.utils.paths import get_project_path

pytestmark = pytest.mark.e2e_smoke


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"row-{index}",
            "meta": {
                "grammar_rev": "rev-1",
                "seed": index,
                "split": "train",
                "template_index": index,
            },
            "task": "lm"
        }
        for index in range(8)
    ]


def test_end_to_end_process_and_manifest_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, _rows())

    out_dir = tmp_path / "out"
    runner = CliRunner()

    result = runner.invoke(
        process_app,
        [
            "run",
            "--in", str(dataset_path),
            "--schema", str(get_project_path("conf/data/sample_dataset.yaml")),
            "--format", "JSONL",
            "--seed", "17",
            "--split", "train=0.75,dev=0.125,test=0.125",
            "--out", str(out_dir),
        ]
    )
    assert result.exit_code == ExitCode.OK.value

    manifest_path = out_dir / "dataset.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["dataset_id"] == dataset_path.stem
    assert set(manifest["splits"].keys()) == {"train", "dev", "test"}
    for split_name, split_meta in manifest["splits"].items():
        assert Path(split_meta["path"]).exists()
        assert split_meta["rows"] >= 0
        assert split_meta["sha256"]

    stats_path = out_dir / "dataset_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["rows_total"] == 8
