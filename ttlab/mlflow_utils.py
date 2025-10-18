"""Utilities wrapping MLflow interactions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

import mlflow

from ttlab.config import MLflowConfig


def ensure_experiment(config: MLflowConfig) -> str:
    """Ensure the configured MLflow experiment exists and return its ID."""

    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment)
    if experiment is not None:
        return experiment.experiment_id  # type: ignore[return-value]

    experiment_id = mlflow.create_experiment(
        name=config.experiment,
        artifact_location=config.artifact_location,
    )
    return experiment_id


def log_dataset_run(
    config: MLflowConfig,
    *,
    run_name: str,
    stats: Dict[str, object],
    params: Optional[Dict[str, object]] = None,
    artifacts: Optional[Dict[str, object]] = None,
) -> str:
    """Create an MLflow run and log dataset statistics and artifacts."""

    ensure_experiment(config)
    mlflow.set_experiment(config.experiment)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_metric("rows", stats.get("rows", 0))
        mlflow.log_metric("columns", stats.get("columns", 0))
        if params:
            mlflow.log_params(params)
        if stats:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix="_dataset_stats.json") as fh:
                json.dump(stats, fh, indent=2)
                stats_path = Path(fh.name)
            mlflow.log_artifact(str(stats_path), artifact_path="dataset")
            stats_path.unlink(missing_ok=True)
        if artifacts:
            for name, payload in artifacts.items():
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=f"_{name}.json") as fh:
                    json.dump(payload, fh, indent=2)
                    artifact_path = Path(fh.name)
                mlflow.log_artifact(str(artifact_path), artifact_path="dataset")
                artifact_path.unlink(missing_ok=True)

        run_id = active_run.info.run_id
    return run_id


def ping_tracking_server(config: MLflowConfig) -> Dict[str, str]:
    """Return metadata about the configured MLflow tracking server."""

    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment)
    status = "existing" if experiment else "created"
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=config.experiment,
            artifact_location=config.artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id  # type: ignore[assignment]
    return {
        "tracking_uri": config.tracking_uri,
        "experiment": config.experiment,
        "experiment_id": experiment_id,
        "status": status,
    }
