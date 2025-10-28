from pathlib import Path

import json
import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from ttlab.cli.cli_process import process_app
from ttlab.utils.paths import get_project_path


pytestmark = pytest.mark.mlflow


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_process_logs_to_mlflow(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    rows = [{"id": "x", "meta": {"split": "train", "template_index": 0}, "task": "lm"}]
    _write_jsonl(data, rows)

    out_dir = tmp_path / "out"
    mlruns = tmp_path / "mlruns"
    uri = f"file://{mlruns.as_posix()}"

    result = CliRunner().invoke(
        process_app,
        [
            "run",
            "--in",
            str(data),
            "--schema",
            str(get_project_path("conf/data/sample_dataset.yaml")),
            "--format",
            "JSONL",
            "--split",
            "train=1.0",
            "--seed",
            "1",
            "--out",
            str(out_dir),
            "--mlflow",
            "--mlflow-uri",
            uri,
            "--experiment",
            "ttlab.dev",
            "--tag",
            "it=1",
        ],
    )
    assert result.exit_code == 0

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("ttlab.dev")
    runs = client.search_runs([experiment.experiment_id], max_results=10)
    assert any("process[" in run.info.run_name for run in runs)
