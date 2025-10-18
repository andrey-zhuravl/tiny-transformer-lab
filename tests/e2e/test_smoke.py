import json
import shutil
from pathlib import Path

import yaml
from typer.testing import CliRunner

from ttlab.cli import app


def test_smoke_command(tmp_path):
    runner = CliRunner()
    base_config = yaml.safe_load(Path("conf/base.yaml").read_text())

    dataset_path = tmp_path / "dataset.jsonl"
    shutil.copy(Path("tests/resources/sample.jsonl"), dataset_path)

    base_config["paths"]["data_raw"] = str(tmp_path / "data_raw")
    base_config["paths"]["data_processed"] = str(tmp_path / "data_processed")
    base_config["paths"]["data_splits"] = str(tmp_path / "data_splits")
    base_config["paths"]["out_metrics"] = str(tmp_path / "out_metrics")
    base_config["paths"]["out_artifacts"] = str(tmp_path / "out_artifacts")
    base_config["data"]["input_path"] = str(dataset_path)
    base_config["mlflow"]["tracking_uri"] = f"file:{(tmp_path / 'mlruns').as_posix()}"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_config))

    result = runner.invoke(app, ["smoke", "--config", str(config_path)])
    assert result.exit_code == 0

    metrics_path = Path(base_config["paths"]["out_metrics"]) / "metrics.jsonl"
    assert metrics_path.exists()
    events = [json.loads(line)["metric"] for line in metrics_path.read_text().splitlines()]
    assert "smoke_start" in events
    assert "smoke_done" in events
