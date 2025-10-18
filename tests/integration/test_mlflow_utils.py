from pathlib import Path

from ttlab.config import MLflowConfig
from ttlab.mlflow_utils import ensure_experiment, log_dataset_run, ping_tracking_server


def test_ensure_experiment_creates(tmp_path):
    tracking_uri = f"file:{tmp_path.as_posix()}"
    config = MLflowConfig(tracking_uri=tracking_uri, experiment="test-exp", artifact_location=None)

    experiment_id = ensure_experiment(config)
    info = ping_tracking_server(config)
    assert info["experiment_id"] == experiment_id

    run_id = log_dataset_run(
        config,
        run_name="demo",
        stats={"rows": 10, "columns": 2},
        params={"format": "PARQUET"},
    )
    assert isinstance(run_id, str)
