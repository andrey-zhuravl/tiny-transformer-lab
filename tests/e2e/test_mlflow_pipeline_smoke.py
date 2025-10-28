from pathlib import Path

import json
import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from ttlab.cli.cli_process import process_app
from ttlab.cli.cli_tokenizer import tokenizer_app
from ttlab.utils.paths import get_project_path


pytestmark = pytest.mark.mlflow


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_end_to_end_logs_nested_runs(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    rows = [
        {"id": str(idx), "meta": {"split": "train", "template_index": idx}, "task": "lm"}
        for idx in range(5)
    ]
    _write_jsonl(data, rows)

    out_dir = tmp_path / "out"
    mlruns = tmp_path / "mlruns"
    uri = f"file://{mlruns.as_posix()}"

    runner = CliRunner()
    process_result = runner.invoke(
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
            "17",
            "--out",
            str(out_dir),
            "--mlflow",
            "--mlflow-uri",
            uri,
            "--experiment",
            "ttlab.dev",
            "--tag",
            "pipeline=1",
        ],
    )
    assert process_result.exit_code == 0

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("ttlab.dev")
    parent_run = client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
    )[0]

    tokenizer_result = runner.invoke(
        tokenizer_app,
        [
            "train",
            "--dataset-manifest",
            str(out_dir / "dataset.manifest.json"),
            "--algo",
            "bpe",
            "--vocab-size",
            "64",
            "--norm",
            "nfc",
            "--lower",
            "--punct-policy",
            "keep",
            "--seed",
            "17",
            "--out",
            str(tmp_path / "tok"),
            "--mlflow",
            "--mlflow-uri",
            uri,
            "--experiment",
            "ttlab.dev",
            "--parent-run-id",
            parent_run.info.run_id,
            "--tag",
            "pipeline=1",
        ],
    )
    assert tokenizer_result.exit_code == 0
