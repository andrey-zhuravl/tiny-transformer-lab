from pathlib import Path

import json
import pytest
from mlflow.tracking import MlflowClient
from typer.testing import CliRunner

from ttlab.cli.cli_tokenizer import tokenizer_app


pytestmark = pytest.mark.mlflow


def _write_manifest(path: Path, dataset_dir: Path) -> None:
    payload = {"splits": {"train": [{"path": str(dataset_dir)}]}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tokenizer_train_logs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.txt").write_text("hello world\n", encoding="utf-8")
    manifest = tmp_path / "dataset.manifest.json"
    _write_manifest(manifest, dataset_dir)

    out_dir = tmp_path / "tok"
    mlruns = tmp_path / "mlruns"
    uri = f"file://{mlruns.as_posix()}"

    result = CliRunner().invoke(
        tokenizer_app,
        [
            "train",
            "--dataset-manifest",
            str(manifest),
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
            "7",
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
    assert any("tokenizer.train[" in run.info.run_name for run in runs)
