import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ttlab.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_init_creates_directories(tmp_path, runner):
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    config = {
        "general": {"seed": 42, "run_name": "cli-test"},
        "paths": {
            "data_raw": str(data_dir / "raw"),
            "data_processed": str(data_dir / "processed"),
            "data_splits": str(data_dir / "splits"),
            "out_metrics": str(tmp_path / "metrics"),
            "out_artifacts": str(tmp_path / "artifacts"),
        },
        "data": {
            "format": "JSONL",
            "input_path": str(tmp_path / "input.jsonl"),
            "schema_fields": [
                {"field_name": "id", "type": "int", "required": True},
                {"field_name": "text", "type": "str", "required": True},
            ],
        },
        "mlflow": {
            "tracking_uri": f"file:{(tmp_path / 'mlruns').as_posix()}",
            "experiment": "cli-test",
            "artifact_location": None,
        },
    }
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    result = runner.invoke(app, ["init", "--config", str(config_path)])
    assert result.exit_code == 0
    assert (data_dir / "raw").exists()
    assert (tmp_path / "metrics").exists()


def test_cli_data_validate(tmp_path, runner):
    sample_jsonl = tmp_path / "sample.jsonl"
    pd.DataFrame({"id": [1], "text": ["ok"]}).to_json(sample_jsonl, orient="records", lines=True)
    schema_path = Path("conf/data/sample_dataset.yaml")

    result = runner.invoke(
        app,
        [
            "data:validate",
            "--in",
            str(sample_jsonl),
            "--schema",
            str(schema_path),
            "--format",
            "JSONL",
        ],
    )
    assert result.exit_code == 0
