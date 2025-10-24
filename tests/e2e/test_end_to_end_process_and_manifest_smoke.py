from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttlab.cli import run
from ttlab.core.validate import ExitCode


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
            "task": {"type": "lm", "text": f"sample text {index}"},
        }
        for index in range(8)
    ]


def test_end_to_end_process_and_manifest_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset_path, _rows())

    out_dir = tmp_path / "out"

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
            "train=0.75",
            "dev=0.125",
            "test=0.125",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code is ExitCode.OK

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
